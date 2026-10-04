"""Repository-wide pytest policy.

- `network` tests (real Hugging Face Hub access) run only with VRAMFORGE_NETWORK_TESTS=1.
- `parity` tests (real TRL/torch trainers) run only when the `parity` dependency group is installed.
- Production modules must never import torch (checked by tests/unit/test_no_torch_import.py).
"""

from __future__ import annotations

import importlib.util
import os

import pytest

# The parity tests build real TRL trainers; never let them send Hub telemetry from test runs.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def _parity_available() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "trl", "peft"))


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    run_network = os.environ.get("VRAMFORGE_NETWORK_TESTS") == "1"
    parity_ok = _parity_available()
    skip_network = pytest.mark.skip(reason="set VRAMFORGE_NETWORK_TESTS=1 to run network tests")
    skip_parity = pytest.mark.skip(
        reason="install the 'parity' dependency group (uv sync --all-groups)"
    )
    for item in items:
        if "network" in item.keywords and not run_network:
            item.add_marker(skip_network)
        if "parity" in item.keywords and not parity_ok:
            item.add_marker(skip_parity)
