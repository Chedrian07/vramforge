"""Fixtures for dataset inspection tests: default access and a `datasets.load_dataset` oracle."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from vramforge_estimator.sources import SourceAccess


@pytest.fixture
def access() -> SourceAccess:
    return SourceAccess()


@pytest.fixture
def datasets_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Callable[..., dict[str, list[dict[str, Any]]]]:
    """What `datasets.load_dataset` (non-streaming, offline) returns, split by split."""
    import datasets

    monkeypatch.setattr(datasets.config, "HF_HUB_OFFLINE", True)
    datasets.disable_progress_bars()
    cache = tmp_path / "datasets-cache"

    def load(path: Path, *args: Any, **kwargs: Any) -> dict[str, list[dict[str, Any]]]:
        if path.is_file():
            loaded = datasets.load_dataset(
                *args, data_files=str(path), cache_dir=str(cache), **kwargs
            )
        else:
            loaded = datasets.load_dataset(str(path), *args, cache_dir=str(cache), **kwargs)
        return {str(name): list(split) for name, split in loaded.items()}

    return load


@pytest.fixture
def fake_hub(monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Serve Hub downloads from local mirror directories: {repo_id: directory}.

    Records every downloaded or remotely opened repo path in `fake_hub["__log__"]`.
    """
    from vramforge_estimator.errors import EstimatorError, make_issue
    from vramforge_estimator.inspection.dataset_files import SourceFiles
    from vramforge_estimator.schemas import ErrorCode

    mirrors: dict[str, Any] = {"__log__": []}

    def local(self: SourceFiles, rel_path: str) -> Path:
        path = mirrors[self.source.repo_id] / rel_path
        mirrors["__log__"].append(rel_path)
        if not path.is_file():
            raise EstimatorError(make_issue(ErrorCode.SOURCE_REVISION_CHANGED, "missing"))
        return path

    def opener(self: SourceFiles, rel_path: str) -> Callable[[], Any]:
        return lambda: local(self, rel_path).open("rb")

    monkeypatch.setattr(SourceFiles, "_download", local)
    monkeypatch.setattr(SourceFiles, "_remote_opener", opener)
    return mirrors
