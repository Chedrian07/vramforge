"""Loads the integration testkit for the security tests (uniquely named module)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


def _load() -> Any:
    name = "integration_testkit"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parents[1] / "integration" / "integration_testkit.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


kit = _load()
