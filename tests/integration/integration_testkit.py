"""Shared helpers for integration and security tests (uniquely named module)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

REPO = Path(__file__).resolve().parents[2]
EXAMPLE_REQUEST = REPO / "tests" / "fixtures" / "requests" / "plan_example_grpo.json"
CSRF = {"X-VramForge-Request": "1"}


def load_pipeline_fakes() -> ModuleType:
    name = "pipeline_fakes"
    if name in sys.modules:
        return sys.modules[name]
    path = REPO / "tests" / "unit" / "pipeline" / "pipeline_fakes.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def request_body(**overrides: Any) -> dict[str, Any]:
    data = json.loads(EXAMPLE_REQUEST.read_text(encoding="utf-8"))
    for dotted, value in overrides.items():
        node = data
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node[key]
        node[leaf] = value
    return data


def ready_grpo_body(**overrides: Any) -> dict[str, Any]:
    """Explicit completion budget and a reward kind: the GRPO combination can be `ready`."""
    return request_body(
        **{"grpo.completion_budget": 1024, "grpo.reward": {"kind": "cpu_rule"}, **overrides}
    )


def read_sse(resp: Any) -> list[dict[str, str]]:
    events: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in resp.iter_lines():
        if not line:
            if current:
                events.append(current)
                current = {}
            continue
        if line.startswith(":"):
            continue
        key, _, value = line.partition(": ")
        current[key] = value
    if current:
        events.append(current)
    return [e for e in events if "id" in e]


def make_settings(tmp_path: Path, monkeypatch: Any, **overrides: Any) -> Any:
    """Settings on SQLite + a temp data dir, mirrored into the environment for the worker."""
    from vramforge_api.db import Base, get_engine
    from vramforge_api.settings import Settings, reset_settings_cache

    values: dict[str, Any] = {
        "database_url": f"sqlite:///{tmp_path / 'it.db'}",
        "redis_url": "redis://fakeredis:6379/0",
        "data_dir": tmp_path / "data",
        "local_roots": f"local={tmp_path / 'local'}",
        "cancel_grace_s": 0,
        "cancel_poll_s": 0.02,
        "sse_poll_interval_s": 0.01,
        "sse_keepalive_s": 0.2,
        "sse_max_duration_s": 10,
        "progress_event_interval_s": 0,
    }
    values.update(overrides)
    for key, value in values.items():
        monkeypatch.setenv(f"VRAMFORGE_{key.upper()}", str(value))
    reset_settings_cache()
    settings = Settings(**values)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(get_engine(settings.database_url))
    return settings
