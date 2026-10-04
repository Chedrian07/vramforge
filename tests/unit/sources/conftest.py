"""Shared helpers for source-resolution tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

FIXTURES = Path(__file__).parents[2] / "fixtures" / "models"


def _load_writer() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "vf_safetensors_writer", FIXTURES / "safetensors_writer.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def st_writer() -> ModuleType:
    return _load_writer()
