"""Shared helpers for source-resolution tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

FIXTURES = Path(__file__).parents[2] / "fixtures" / "models"


def load_fixture_module(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"vf_fixture_{name}", FIXTURES / f"{name}.py")
    assert spec is not None and spec.loader is not None
    if spec.name in sys.modules:
        return sys.modules[spec.name]
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def st_writer() -> ModuleType:
    return load_fixture_module("safetensors_writer")


@pytest.fixture(scope="session")
def fake_hub() -> ModuleType:
    return load_fixture_module("fake_hub")
