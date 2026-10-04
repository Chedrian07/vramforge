"""TokenizerInspector: the model's own tokenizer and chat template (plan.md §6.2, §7.3).

Only tokenizer/template files (plus config.json, whose ``model_type`` selects the tokenizer class
in transformers 5.18 — docs/research/example-model-dataset.md §1) are fetched at the pinned
commit, checked against the manifest and copied into a private staging directory. The tokenizer
is built from exactly those files with ``trust_remote_code=False``; another model's tokenizer or
template is never substituted. Loading the staged copy keeps the repo id as the config's
``name_or_path`` so class selection matches loading by id.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from vramforge_estimator.errors import EstimatorError, make_issue
from vramforge_estimator.schemas import ErrorCode, Stage, TokenizerManifest
from vramforge_estimator.sources import ResolvedSource, SourceAccess

from .base import TokenizerHandle
from .model_config import load_config
from .model_files import SourceFiles, open_source_files
from .tokenizer_manifest import (
    has_generation_markers,
    max_length,
    special_token_defaults,
    template_kwargs,
    template_sha256,
    tokenizer_fingerprint,
)

TOKENIZER_FILES = frozenset(
    {
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "tokenizer.model",
        "chat_template.jinja",
        "vocab.txt",
        "spiece.model",
        "sentencepiece.bpe.model",
        "tekken.json",
        "tiktoken.model",
    }
)
VOCAB_FILES = frozenset(
    {
        "tokenizer.json",
        "tokenizer.model",
        "vocab.json",
        "vocab.txt",
        "spiece.model",
        "sentencepiece.bpe.model",
        "tekken.json",
        "tiktoken.model",
    }
)
TEMPLATE_DIR = "additional_chat_templates"
# Read only to find a template held by the processor (legacy multimodal repos).
PROCESSOR_TEMPLATE_FILES = ("chat_template.json", "processor_config.json")

TemplateSource = Literal["chat_template.jinja", "tokenizer_config.json", "processor", "none"]


def _error(code: ErrorCode, message: str, **details: object) -> EstimatorError:
    return EstimatorError(
        make_issue(code, message, stage=Stage.INSPECTING, component="tokenizer", **details)
    )


def _json_object(data: bytes | None) -> dict[str, Any]:
    if data is None:
        return {}
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _text(data: bytes) -> str:
    # transformers reads template files in text mode (universal newlines)
    return data.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")


def tokenizer_file_names(paths: set[str], tokenizer_config: dict[str, Any]) -> list[str]:
    names = {p for p in paths if p in TOKENIZER_FILES}
    for path in paths:
        parent = PurePosixPath(path).parent.as_posix()
        if parent == TEMPLATE_DIR and path.endswith(".jinja"):
            names.add(path)
    fast_files = tokenizer_config.get("fast_tokenizer_files")
    if isinstance(fast_files, list):
        names |= {f for f in fast_files if isinstance(f, str) and f in paths and "/" not in f}
    return sorted(names)


def _requires_remote_code(tokenizer_config: dict[str, Any], model_type: str | None) -> bool:
    """A custom tokenizer class from the repo with no native transformers class (plan §18)."""
    auto_map = tokenizer_config.get("auto_map")
    refs = auto_map if isinstance(auto_map, list | tuple) else None
    if isinstance(auto_map, dict):
        refs = auto_map.get("AutoTokenizer")
    if not refs:
        return False
    from transformers.models.auto.tokenization_auto import (
        MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS,
        tokenizer_class_from_name,
    )

    if model_type in MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS:
        return False  # transformers ignores the remote class without trust_remote_code
    name = tokenizer_config.get("tokenizer_class")
    if not isinstance(name, str):
        return True
    return (
        tokenizer_class_from_name(name) is None
        and tokenizer_class_from_name(name.removesuffix("Fast")) is None
    )


def _processor_template(files: SourceFiles, paths: set[str]) -> tuple[str, str] | None:
    """(template, file name) of a template stored only with the processor."""
    for name in PROCESSOR_TEMPLATE_FILES:
        if name in paths:
            template = _json_object(files.read(name)).get("chat_template")
            if isinstance(template, str) and template:
                return template, name
    return None


def _template_source(
    template: str, staged: dict[str, bytes], tokenizer_config: dict[str, Any]
) -> TemplateSource:
    jinja = staged.get("chat_template.jinja")
    if jinja is not None and _text(jinja) == template:
        return "chat_template.jinja"
    configured = tokenizer_config.get("chat_template")
    if isinstance(configured, list):
        named = {t.get("name"): t.get("template") for t in configured if isinstance(t, dict)}
        configured = named.get("default")
    if configured == template:
        return "tokenizer_config.json"
    if any(name.endswith(".jinja") for name in staged):
        return "chat_template.jinja"
    return "tokenizer_config.json"


def _load(stage: Path, *, has_config: bool, repo_id: str | None) -> Any:
    from transformers import AutoConfig, AutoTokenizer, PreTrainedConfig

    config: Any = None
    if has_config:
        try:
            config = AutoConfig.from_pretrained(stage, trust_remote_code=False)
        except (ValueError, OSError):
            config = None
    if config is None:
        config = PreTrainedConfig.from_pretrained(stage)
    if repo_id is not None:
        config.name_or_path = repo_id  # same class selection as AutoTokenizer.from_pretrained(id)
    return AutoTokenizer.from_pretrained(
        stage, config=config, trust_remote_code=False, local_files_only=True
    )


def load_tokenizer(source: ResolvedSource, access: SourceAccess) -> TokenizerHandle:
    """Load the model's own tokenizer and chat template (trust_remote_code=False).

    Raises `EstimatorError` with TOKENIZER_REQUIRED / TEMPLATE_REQUIRED / REMOTE_CODE_REQUIRED.
    Never substitutes another model's tokenizer or template. A missing template is reported as
    ``chat_template_present=False``; the pipeline decides whether that blocks the objective.
    """
    files = open_source_files(source, access)
    paths = set(files.entries())
    tokenizer_config = _json_object(
        files.read("tokenizer_config.json") if "tokenizer_config.json" in paths else None
    )
    names = tokenizer_file_names(paths, tokenizer_config)
    if not VOCAB_FILES & set(names):
        raise _error(
            ErrorCode.TOKENIZER_REQUIRED,
            "모델 source에 tokenizer 파일(tokenizer.json, tokenizer.model, vocab 파일)이 없습니다. "
            "다른 모델의 tokenizer로 대체하지 않으므로 tokenizer가 포함된 모델을 사용하세요.",
            reason="tokenizer_files_missing",
        )
    config_bytes = files.read("config.json") if "config.json" in paths else None
    config = load_config(config_bytes) if config_bytes is not None else None
    model_type = config.model_type if config is not None else None
    if _requires_remote_code(tokenizer_config, model_type):
        raise _error(
            ErrorCode.REMOTE_CODE_REQUIRED,
            "이 tokenizer는 저장소에 포함된 사용자 코드가 있어야 로드됩니다. "
            "원격 코드는 실행하지 않으므로 지원하지 않습니다.",
            reason="custom_tokenizer_class",
        )

    staged = {name: files.read(name) for name in names}
    processor = None
    with tempfile.TemporaryDirectory(prefix="vf-tokenizer-") as tmp:
        stage = Path(tmp)
        for name, data in staged.items():
            (stage / name).parent.mkdir(parents=True, exist_ok=True)
            (stage / name).write_bytes(data)
        if config_bytes is not None:
            (stage / "config.json").write_bytes(config_bytes)
        try:
            tokenizer = _load(stage, has_config=config_bytes is not None, repo_id=source.repo_id)
        except Exception as exc:
            raise _error(
                ErrorCode.TOKENIZER_REQUIRED,
                "모델의 tokenizer를 불러오지 못했습니다. tokenizer 파일 형식을 확인하세요.",
                reason="tokenizer_load_failed",
                error_type=type(exc).__name__,
            ) from None

    # never keep the private staging path on the handle (results must not carry host paths)
    tokenizer.name_or_path = source.repo_id or source.manifest.reference
    template = tokenizer.chat_template
    if isinstance(template, dict):
        template = template.get("default")
    if not template:
        processor = _processor_template(files, paths)
        if processor is not None:
            template = processor[0]
            tokenizer.chat_template = template  # same repository, held by the processor

    files_sha256 = {name: hashlib.sha256(data).hexdigest() for name, data in staged.items()}
    source_name: TemplateSource = "none"
    sha: str | None = None
    kwargs: list[str] = []
    if template:
        sha = template_sha256(template)
        if processor is not None:
            source_name = "processor"
            files_sha256[processor[1]] = hashlib.sha256(files.read(processor[1])).hexdigest()
        else:
            source_name = _template_source(template, staged, tokenizer_config)
        kwargs = template_kwargs(template, getattr(tokenizer, "special_tokens_map", {}) or {}) or []

    adds_bos, adds_eos = special_token_defaults(tokenizer)
    limit, sentinel = max_length(getattr(tokenizer, "model_max_length", None))
    tokenizer_class = type(tokenizer).__name__
    config_vocab = None
    if config is not None:
        value = config.text.get("vocab_size")
        config_vocab = value if isinstance(value, int) and not isinstance(value, bool) else None

    manifest = TokenizerManifest(
        tokenizer_class=tokenizer_class,
        vocab_size=len(tokenizer),
        config_vocab_size=config_vocab,
        bos_token=_token(tokenizer, "bos_token"),
        eos_token=_token(tokenizer, "eos_token"),
        pad_token=_token(tokenizer, "pad_token"),
        adds_bos_by_default=adds_bos,
        adds_eos_by_default=adds_eos,
        model_max_length=limit,
        model_max_length_is_sentinel=sentinel,
        chat_template_present=bool(template),
        chat_template_source=source_name,
        chat_template_sha256=sha,
        has_generation_markers=bool(template) and has_generation_markers(template),
        template_kwargs=kwargs,
        files_sha256=files_sha256,
        fingerprint=tokenizer_fingerprint(
            tokenizer_class=tokenizer_class,
            files_sha256=files_sha256,
            chat_template_sha256=sha,
            model_type=model_type,
        ),
    )
    return TokenizerHandle(tokenizer=tokenizer, manifest=manifest)


def _token(tokenizer: Any, name: str) -> str | None:
    value = getattr(tokenizer, name, None)
    return str(value) if value is not None else None
