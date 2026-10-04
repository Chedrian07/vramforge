"""Shared fixtures for the architecture adapter tests (offline, no torch)."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest
from arch_helpers import make_cfg

from vramforge_estimator.schemas import ModelInventory, ResolvedConfig

BUILDER_PATH = (
    Path(__file__).resolve().parents[2] / "fixtures" / "inventories" / "inventory_builder.py"
)


def _load_builder() -> ModuleType:
    name = "vf_inventory_builder"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, BUILDER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def ib() -> ModuleType:
    return _load_builder()


@pytest.fixture(scope="session")
def mimo(ib: ModuleType) -> ModelInventory:
    return ib.load_mimo_inventory()


@pytest.fixture
def cfg_factory() -> Callable[..., ResolvedConfig]:
    return make_cfg
