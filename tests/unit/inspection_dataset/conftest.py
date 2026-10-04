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
