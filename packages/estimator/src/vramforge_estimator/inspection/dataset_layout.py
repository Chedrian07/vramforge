"""Configs, splits and data files of a dataset source, resolved exactly like datasets==5.0.1.

Mirrors `datasets.load.HubDatasetModuleFactory.get_module` / `LocalDatasetModuleFactory.get_module`
plus `BuilderConfig._resolve_data_files`, reusing datasets' own helpers (README `configs` via
`MetadataConfigs`, data-file pattern inference via `get_data_patterns`/`resolve_pattern`, builder
module inference, `create_builder_configs_from_metadata_configs`). Split membership and file order
therefore equal what a trainer's `load_dataset(repo_id, revision=sha)` or `load_dataset(dir)` sees:
files sorted per pattern, patterns in datasets' order.

For Hub repos the directory listing comes from the pinned manifest (the resolver's file list)
through an in-memory fsspec filesystem, so resolution is offline and deterministic. Only card files
(README.md, .huggingface.yaml, dataset_infos.json) are downloaded here. Repos that ship a loading
script are rejected, like datasets 5 does; no code from the dataset is ever executed.

README `configs` may only point inside the dataset (plan §18): datasets would follow absolute
paths, `..`, URLs and other repos (`hf://`) in `data_files` / `data_dir`, which here would glob the
worker's filesystem, reach arbitrary hosts or list other Hub repos with the server's credentials.
Such configs are refused before anything is resolved.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from fsspec import AbstractFileSystem

from vramforge_estimator.errors import EstimatorError, make_issue
from vramforge_estimator.schemas import ErrorCode, Issue, Severity, Stage

from .dataset_files import DataFile, SourceFiles, SourceKind
from .readers import SUPPORTED_MODULES

if TYPE_CHECKING:
    from datasets import DownloadConfig, Features
    from huggingface_hub import DatasetCardData

logger = logging.getLogger(__name__)
_PROTOCOL = "vfmanifest"
_ROOT = "repo"
_TREES: dict[str, dict[str, int]] = {}
_REGISTER_LOCK = threading.Lock()
_registered = False
# BuilderConfig fields that describe where data lives, not how it is read.
_BASE_FIELDS = frozenset({"name", "version", "data_dir", "data_files", "description"})
_ARCHIVE_SUFFIXES = (".zip", ".tar", ".tgz", ".tar.gz", ".tar.bz2", ".tar.xz", ".7z", ".rar")
_MAX_NOTE_EXAMPLES = 5


@dataclass(frozen=True)
class SplitLayout:
    name: str
    files: tuple[DataFile, ...]
    num_rows: int | None = None  # README dataset_info (datasets verifies it when loading)
    num_bytes: int | None = None


@dataclass(frozen=True)
class ConfigLayout:
    name: str
    module: str | None  # datasets packaged builder: json | csv | parquet | arrow | ...
    options: Mapping[str, Any] = field(default_factory=dict)  # builder params that shape reading
    splits: tuple[SplitLayout, ...] = ()
    unsupported: Issue | None = None
    # README `dataset_info` features: datasets builds the builder with this info, so every table
    # of every split is cast to them (`_cast_table`, ArrowWriter) instead of an inferred schema.
    features: Features | None = None

    def split(self, name: str) -> SplitLayout | None:
        return next((split for split in self.splits if split.name == name), None)


@dataclass(frozen=True)
class DatasetLayout:
    kind: SourceKind
    configs: tuple[ConfigLayout, ...]
    default_config: str | None = None
    notes: tuple[str, ...] = ()  # Korean, display-safe

    def config(self, name: str) -> ConfigLayout | None:
        return next((config for config in self.configs if config.name == name), None)


class ManifestFileSystem(AbstractFileSystem):
    """Read-only directory listing of a pinned file list (no file contents)."""

    protocol = _PROTOCOL
    root_marker = ""
    cachable = False

    def __init__(self, tree: str = "", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        listing = _TREES.get(tree)
        if listing is None:
            raise FileNotFoundError("unknown manifest listing")
        self._files = {f"{_ROOT}/{path}": size for path, size in listing.items()}
        self._dirs: set[str] = {_ROOT}
        self._children: dict[str, set[str]] = {}
        for path in self._files:
            parts = path.split("/")
            for i in range(1, len(parts)):
                parent, child = "/".join(parts[:i]), "/".join(parts[: i + 1])
                self._dirs.add(parent)
                self._children.setdefault(parent, set()).add(child)

    def _entry(self, path: str) -> dict[str, Any]:
        if path in self._files:
            return {"name": path, "size": self._files[path], "type": "file"}
        if path in self._dirs:
            return {"name": path, "size": 0, "type": "directory"}
        raise FileNotFoundError(path)

    def info(self, path: str, **kwargs: Any) -> dict[str, Any]:
        return self._entry(self._strip_protocol(path))

    def ls(self, path: str, detail: bool = True, **kwargs: Any) -> list[Any]:
        path = self._strip_protocol(path)
        if path in self._files:
            entries = [self._entry(path)]
        elif path in self._dirs:
            entries = [self._entry(child) for child in sorted(self._children.get(path, ()))]
        else:
            raise FileNotFoundError(path)
        return entries if detail else [entry["name"] for entry in entries]

    def _open(self, path: str, mode: str = "rb", **kwargs: Any) -> Any:
        raise NotImplementedError("manifest listings have no file contents")


def _register_filesystem() -> None:
    global _registered
    if _registered:
        return
    with _REGISTER_LOCK:
        if not _registered:
            import fsspec

            fsspec.register_implementation(_PROTOCOL, ManifestFileSystem, clobber=True)
            _registered = True


@dataclass(frozen=True)
class _Listing:
    base_path: str
    download_config: DownloadConfig | None
    to_rel: Callable[[str], str]
    # Whether a resolved URL lies in the pinned listing (local paths are checked on disk).
    inside: Callable[[str], bool] = lambda url: True


@contextmanager
def _listing(files: SourceFiles) -> Iterator[_Listing]:
    if files.kind == "hf":
        from datasets import DownloadConfig

        _register_filesystem()
        key = uuid.uuid4().hex
        _TREES[key] = {entry.path: entry.size or 0 for entry in files.source.manifest.files}
        prefix = f"{_PROTOCOL}://{_ROOT}/"
        try:
            yield _Listing(
                base_path=f"{_PROTOCOL}://{_ROOT}",
                download_config=DownloadConfig(storage_options={_PROTOCOL: {"tree": key}}),
                to_rel=lambda url: url.removeprefix(prefix),
                inside=lambda url: url.startswith(prefix),
            )
        finally:
            _TREES.pop(key, None)
    else:
        assert files.root is not None
        base = files.root.as_posix()
        yield _Listing(
            base_path=base,
            download_config=None,
            to_rel=lambda path: PurePosixPath(os.path.relpath(path, base)).as_posix(),
        )


def resolve_layout(files: SourceFiles) -> DatasetLayout:
    """Configs → splits → ordered data files. Raises `EstimatorError` when nothing is loadable."""
    try:
        if files.kind == "local_file":
            return _single_file_layout(files)
        _reject_loading_script(files)
        card_data = _load_card(files)
        with _listing(files) as listing:
            return _module_layout(files, listing, card_data)
    except EstimatorError:
        raise
    except Exception as exc:  # a datasets helper failed in an unexpected way: never a raw error
        logger.error("unexpected %s while resolving the dataset layout", type(exc).__name__)
        raise EstimatorError(
            _issue(
                ErrorCode.INTERNAL_ERROR,
                "데이터셋 구성(config·split)을 확인하는 중 예기치 못한 오류가 발생했습니다.",
                reason="layout_unexpected_error",
                error_type=type(exc).__name__,
            )
        ) from exc


def _module_layout(
    files: SourceFiles, listing: _Listing, card_data: DatasetCardData
) -> DatasetLayout:
    from datasets.data_files import EmptyDatasetError, get_data_patterns, sanitize_patterns
    from datasets.info import DatasetInfosDict
    from datasets.load import (
        create_builder_configs_from_metadata_configs,
        infer_module_for_data_files,
    )
    from datasets.packaged_modules import (
        _ALL_ALLOWED_EXTENSIONS,
        _MODULE_TO_EXTENSIONS,
        _MODULE_TO_METADATA_EXTENSIONS,
        _MODULE_TO_METADATA_FILE_NAMES,
        _PACKAGED_DATASETS_MODULES,
    )
    from datasets.utils.metadata import MetadataConfigs

    dl = listing.download_config
    try:
        metadata_configs = MetadataConfigs.from_dataset_card_data(card_data)
        dataset_infos = DatasetInfosDict.from_dataset_card_data(card_data)
        _check_configs_stay_inside(metadata_configs)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise EstimatorError(_unsupported("readme_configs_invalid")) from exc
    try:
        if metadata_configs and "data_files" in next(iter(metadata_configs.values())):
            patterns = sanitize_patterns(next(iter(metadata_configs.values()))["data_files"])
        else:
            patterns = get_data_patterns(listing.base_path, download_config=dl)
        data_files = {
            str(split): _resolve_patterns(pats, [_ALL_ALLOWED_EXTENSIONS] * len(pats), listing)
            for split, pats in patterns.items()
        }
    except EmptyDatasetError:
        raise EstimatorError(_no_data_files(files)) from None
    except (FileNotFoundError, ValueError) as exc:
        raise EstimatorError(_unsupported("data_files_unresolvable")) from exc
    if not any(data_files.values()):
        raise EstimatorError(_no_data_files(files))
    try:
        module_name, default_kwargs = infer_module_for_data_files(
            data_files, path=files.source.manifest.reference, download_config=dl
        )
    except Exception as exc:  # datasets raises several types here (mixed formats, archives)
        raise EstimatorError(_unsupported("module_inference_failed")) from exc
    module_path, _ = _PACKAGED_DATASETS_MODULES[module_name]
    keep_extensions = (
        _MODULE_TO_EXTENSIONS[module_name] + _MODULE_TO_METADATA_EXTENSIONS[module_name]
    )
    keep_names = _MODULE_TO_METADATA_FILE_NAMES[module_name]

    raw_configs: list[tuple[str, dict[str, list[str]] | None, dict[str, Any]]] = []
    default_config: str | None
    if metadata_configs:
        try:
            builder_configs, default_config = create_builder_configs_from_metadata_configs(
                module_path,
                metadata_configs,
                base_path=listing.base_path,
                default_builder_kwargs=default_kwargs,
                download_config=dl,
            )
        except EmptyDatasetError:
            raise EstimatorError(_no_data_files(files)) from None
        except (ValueError, TypeError, FileNotFoundError) as exc:
            raise EstimatorError(_unsupported("readme_configs_invalid")) from exc
        for builder_config in builder_configs:
            resolved: dict[str, list[str]] | None
            try:
                resolved = _resolve_builder_config_files(builder_config, listing)
            except (FileNotFoundError, ValueError):
                resolved = None
            raw_configs.append((builder_config.name, resolved, _options(builder_config)))
    else:
        filtered = {
            split: _filter_files(urls, keep_extensions, keep_names)
            for split, urls in data_files.items()
        }
        raw_configs.append(("default", filtered, dict(default_kwargs)))
        default_config = None

    legacy = _legacy_infos(files)
    if legacy is not None:
        legacy.update(dataset_infos)
        dataset_infos = legacy
    if default_config is None and len(dataset_infos) == 1:
        default_config = next(iter(dataset_infos))

    configs = tuple(
        _config_layout(files, listing, name, module_name, options, resolved, dataset_infos)
        for name, resolved, options in raw_configs
    )
    notes = _layout_notes(listing, configs, _ALL_ALLOWED_EXTENSIONS)
    return DatasetLayout(
        kind=files.kind, configs=configs, default_config=default_config, notes=notes
    )


def _config_layout(
    files: SourceFiles,
    listing: _Listing,
    name: str,
    module: str,
    options: dict[str, Any],
    resolved: dict[str, list[str]] | None,
    dataset_infos: Mapping[str, Any],
) -> ConfigLayout:
    from datasets.packaged_modules import _MODULE_TO_EXTENSIONS

    if resolved is None:
        return ConfigLayout(
            name=name,
            module=module,
            options=options,
            unsupported=_unsupported("config_files_missing"),
        )
    info = dataset_infos.get(name)
    split_infos = getattr(info, "splits", None) or {}
    splits: list[SplitLayout] = []
    for split_name, urls in resolved.items():
        data_files = tuple(_data_file(files, listing, url) for url in urls)
        split_info = split_infos.get(split_name)
        splits.append(
            SplitLayout(
                name=split_name,
                files=data_files,
                num_rows=getattr(split_info, "num_examples", None),
                num_bytes=getattr(split_info, "num_bytes", None),
            )
        )
    unsupported = None
    all_files = [data_file.shard_id for split in splits for data_file in split.files]
    if module not in SUPPORTED_MODULES:
        unsupported = _unsupported("module_not_supported", module=module)
    elif any(path.lower().endswith(_ARCHIVE_SUFFIXES) for path in all_files):
        unsupported = _unsupported("archive_not_supported")
    elif _filter_files(all_files, _MODULE_TO_EXTENSIONS[module], []) != all_files:
        unsupported = _unsupported("mixed_file_formats", module=module)
    elif options.get("newlines_in_values") is not None or options.get("filters") is not None:
        unsupported = _unsupported("builder_option_not_supported")
    return ConfigLayout(
        name=name,
        module=module,
        options=options,
        splits=tuple(splits),
        unsupported=unsupported,
        features=getattr(info, "features", None),
    )


def _data_file(files: SourceFiles, listing: _Listing, url: str) -> DataFile:
    shard_id = listing.to_rel(url)
    entry = files.index.find(shard_id)
    if files.kind == "hf":
        if not listing.inside(url) or not files.has_repo_file(shard_id):
            raise EstimatorError(_unsupported("data_files_outside_dataset"))
        return DataFile(
            shard_id=shard_id, location=shard_id, size=entry.size if entry else None, entry=entry
        )
    path = Path(url)
    files.check_inside(path, shard_id)
    return DataFile(shard_id=shard_id, location=str(path), size=path.stat().st_size, entry=entry)


def _check_configs_stay_inside(metadata_configs: Mapping[str, Mapping[str, Any]]) -> None:
    """Refuse README `configs` whose `data_files` / `data_dir` leave the dataset (module doc)."""
    from datasets.data_files import sanitize_patterns

    for params in metadata_configs.values():
        values: list[Any] = [params.get("data_dir")]
        if params.get("data_files") is not None:
            for patterns in sanitize_patterns(params["data_files"]).values():
                values.extend(patterns)
        if not all(_inside_dataset(value) for value in values):
            raise EstimatorError(_unsupported("data_files_outside_dataset"))


def _inside_dataset(value: Any) -> bool:
    """A dataset-relative path or glob: no scheme or fsspec hop, not absolute, no `..`."""
    from datasets.utils.file_utils import is_relative_path

    if value is None or value == "":
        return True  # datasets falls back to the dataset root
    if not isinstance(value, str):
        return False
    text = value.replace("\\", "/")
    return (
        is_relative_path(value)
        and "::" not in text
        and not text.startswith("/")
        and ".." not in text.split("/")
    )


def _resolve_patterns(
    patterns: list[str], allowed: list[list[str] | None], listing: _Listing
) -> list[str]:
    """`DataFilesList.from_patterns` / `DataFilesPatternsList.resolve` without origin metadata."""
    from glob import has_magic

    from datasets.data_files import resolve_pattern

    out: list[str] = []
    for pattern, extensions in zip(patterns, allowed, strict=True):
        try:
            out.extend(
                resolve_pattern(
                    pattern,
                    base_path=listing.base_path,
                    allowed_extensions=extensions,
                    download_config=listing.download_config,
                )
            )
        except FileNotFoundError:
            if not has_magic(pattern):
                raise
    return out


def _resolve_builder_config_files(builder_config: Any, listing: _Listing) -> dict[str, list[str]]:
    """`BuilderConfig._resolve_data_files` (data_dir joined to the base path)."""
    from datasets.data_files import DataFilesPatternsDict
    from datasets.utils.file_utils import xjoin

    data_files = builder_config.data_files
    if not isinstance(data_files, DataFilesPatternsDict):
        return {str(split): list(urls) for split, urls in (data_files or {}).items()}
    base = xjoin(listing.base_path, builder_config.data_dir) if builder_config.data_dir else None
    scoped = (
        listing
        if base is None
        else _Listing(base, listing.download_config, listing.to_rel, listing.inside)
    )
    return {
        str(split): _resolve_patterns(list(patterns), list(patterns.allowed_extensions), scoped)
        for split, patterns in data_files.items()
    }


def _filter_files(urls: list[str], extensions: list[str], file_names: list[str]) -> list[str]:
    """`DataFilesList.filter` (extension may be followed by one more suffix, e.g. `.gz`)."""
    import re

    patterns = []
    if extensions:
        ext_pattern = "|".join(re.escape(ext) for ext in extensions)
        patterns.append(re.compile(f".*({ext_pattern})(\\..+)?$"))
    if file_names:
        name_pattern = "|".join(re.escape(name) for name in file_names)
        patterns.append(re.compile(rf".*[\/]?({name_pattern})$"))
    if not patterns:
        return list(urls)
    return [url for url in urls if any(pattern.match(url) for pattern in patterns)]


def _options(builder_config: Any) -> dict[str, Any]:
    return {
        f.name: getattr(builder_config, f.name)
        for f in dataclass_fields(builder_config)
        if f.name not in _BASE_FIELDS
    }


def _single_file_layout(files: SourceFiles) -> DatasetLayout:
    """`load_dataset(<module>, data_files=<file>)`: one "train" split of one file."""
    from datasets.load import infer_module_for_data_files_list

    assert files.root is not None
    entries = list(files.source.manifest.files)
    display = PurePosixPath(entries[0].path).name if len(entries) == 1 else files.root.name
    module_name = None
    options: dict[str, Any] = {}
    for logical in (files.root.name, display):
        if logical.lower().endswith(".py"):
            raise EstimatorError(_unsupported("loading_script"))
        module_name, options = infer_module_for_data_files_list([logical])
        if module_name is not None:
            break
    if module_name is None:
        raise EstimatorError(_unsupported("unknown_file_type"))
    entry = entries[0] if len(entries) == 1 else files.index.find(display)
    data_file = DataFile(
        shard_id=display, location=str(files.root), size=files.root.stat().st_size, entry=entry
    )
    unsupported = None
    if module_name not in SUPPORTED_MODULES:
        unsupported = _unsupported("module_not_supported", module=module_name)
    elif display.lower().endswith(_ARCHIVE_SUFFIXES):
        unsupported = _unsupported("archive_not_supported")
    config = ConfigLayout(
        name="default",
        module=module_name,
        options=dict(options),
        splits=(SplitLayout(name="train", files=(data_file,)),),
        unsupported=unsupported,
    )
    return DatasetLayout(kind="local_file", configs=(config,), default_config=None)


def _reject_loading_script(files: SourceFiles) -> None:
    """datasets 5 refuses `<name>/<name>.py`; we never run dataset code either."""
    name = (
        files.source.repo_id.split("/")[-1]
        if files.source.repo_id is not None
        else files.root.name
        if files.root is not None
        else ""
    )
    if name and files.has_repo_file(f"{name}.py"):
        raise EstimatorError(_unsupported("loading_script"))


def _load_card(files: SourceFiles) -> DatasetCardData:
    import yaml
    from huggingface_hub import DatasetCard, DatasetCardData

    try:
        readme = files.metadata_path("README.md")
        card_data = DatasetCard.load(readme).data if readme is not None else DatasetCardData()
        standalone = files.metadata_path(".huggingface.yaml")
        if standalone is not None:
            extra = yaml.safe_load(standalone.read_text(encoding="utf-8"))
            if extra:
                merged = card_data.to_dict()
                merged.update(extra)
                card_data = DatasetCardData(**merged)
    except EstimatorError:
        raise
    except (ValueError, TypeError, yaml.YAMLError, UnicodeDecodeError) as exc:
        raise EstimatorError(_unsupported("readme_yaml_invalid")) from exc
    return card_data


def _legacy_infos(files: SourceFiles) -> Any:
    """Deprecated `dataset_infos.json` from old push_to_hub versions, merged like datasets."""
    from datasets.info import DatasetInfo, DatasetInfosDict

    path = files.metadata_path("dataset_infos.json")
    if path is None:
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        legacy = DatasetInfosDict({name: DatasetInfo.from_dict(info) for name, info in raw.items()})
    except (ValueError, TypeError, AttributeError) as exc:
        raise EstimatorError(_unsupported("dataset_infos_invalid")) from exc
    if len(legacy) == 1:
        legacy["default"] = legacy.pop(next(iter(legacy)))
    return legacy


def _layout_notes(
    listing: _Listing, configs: tuple[ConfigLayout, ...], allowed: list[str]
) -> tuple[str, ...]:
    from datasets.data_files import resolve_pattern

    assigned = [f.shard_id for config in configs for split in config.splits for f in split.files]
    notes: list[str] = []
    try:
        candidates = [
            listing.to_rel(url)
            for url in resolve_pattern(
                "**",
                base_path=listing.base_path,
                allowed_extensions=allowed,
                download_config=listing.download_config,
            )
        ]
    except FileNotFoundError:
        candidates = []
    unassigned = sorted(set(candidates) - set(assigned))
    if unassigned:
        examples = ", ".join(unassigned[:_MAX_NOTE_EXAMPLES])
        notes.append(
            f"어떤 split에도 속하지 않는 데이터 파일 {len(unassigned)}개는 datasets 규칙에 따라 "
            f"분석에서 제외됩니다: {examples}"
        )
    for config in configs:
        for split in config.splits:
            ids = [f.shard_id for f in split.files]
            if len(ids) != len(set(ids)):
                notes.append(
                    f"{config.name}/{split.name} split에는 같은 파일이 여러 번 포함되어 있어 "
                    "datasets와 마찬가지로 중복해서 읽습니다."
                )
    return tuple(notes)


def _issue(code: ErrorCode, message: str, **details: Any) -> Issue:
    return make_issue(
        code,
        message,
        severity=Severity.ERROR,
        stage=Stage.INSPECTING,
        component="dataset",
        **details,
    )


_UNSUPPORTED_MESSAGES = {
    "loading_script": "데이터셋 로딩 스크립트(.py)가 있는 저장소는 지원하지 않습니다. "
    "스크립트는 실행하지 않습니다.",
    "readme_configs_invalid": "README의 configs 설정을 해석할 수 없습니다.",
    "readme_yaml_invalid": "README 메타데이터(YAML)를 해석할 수 없습니다.",
    "dataset_infos_invalid": "dataset_infos.json을 해석할 수 없습니다.",
    "data_files_unresolvable": "데이터 파일 목록을 확정할 수 없습니다.",
    "data_files_outside_dataset": "README의 configs가 데이터셋 밖(절대 경로·상위 경로·URL·다른 "
    "저장소)을 가리켜 읽지 않았습니다. 데이터셋 안의 상대 경로만 지원합니다.",
    "module_inference_failed": "데이터 파일 형식을 판별할 수 없거나 split마다 형식이 다릅니다.",
    "module_not_supported": "지원하지 않는 데이터 형식입니다. JSON/JSONL/Parquet/Arrow/CSV만 "
    "분석할 수 있습니다.",
    "archive_not_supported": "압축 아카이브(zip/tar 등) 안의 데이터는 지원하지 않습니다.",
    "mixed_file_formats": "한 설정 안에 서로 다른 형식의 데이터 파일이 섞여 있습니다.",
    "builder_option_not_supported": "README 설정의 일부 옵션(row filter 등)은 지원하지 않습니다.",
    "config_files_missing": "설정에 지정된 데이터 파일을 찾을 수 없습니다.",
    "unknown_file_type": "파일 확장자로 데이터 형식을 알 수 없습니다. "
    ".json/.jsonl/.parquet/.arrow/.csv 파일을 사용해 주세요.",
}


def _unsupported(reason: str, **details: Any) -> Issue:
    return _issue(
        ErrorCode.DATASET_FORMAT_UNSUPPORTED,
        _UNSUPPORTED_MESSAGES[reason],
        reason=reason,
        **details,
    )


def _no_data_files(files: SourceFiles) -> Issue:
    has_script = any(
        PurePosixPath(entry.path).suffix == ".py" for entry in files.source.manifest.files
    ) or (files.root is not None and files.kind == "local_dir" and any(files.root.glob("*.py")))
    if has_script:
        return _unsupported("loading_script")
    return _issue(
        ErrorCode.SOURCE_NOT_FOUND,
        "데이터셋에서 읽을 수 있는 데이터 파일을 찾지 못했습니다.",
        reason="no_data_files",
    )
