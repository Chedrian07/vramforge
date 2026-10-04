"""Helpers for worker tests (uniquely named: test directories are not packages)."""

from __future__ import annotations

import importlib.util
import sys
from datetime import timedelta
from pathlib import Path
from types import ModuleType
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from vramforge_api import store
from vramforge_api.db import session_scope, utcnow
from vramforge_api.models import Analysis

OWNER = "0" * 64
OTHER_OWNER = "1" * 64


def load_pipeline_fakes() -> ModuleType:
    """tests/unit/pipeline/pipeline_fakes.py, shared by pipeline, worker and API tests."""
    name = "pipeline_fakes"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parents[1] / "pipeline" / "pipeline_fakes.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def create_analysis(sessions: sessionmaker[Session], owner: str = OWNER, **values: Any) -> str:
    fakes = load_pipeline_fakes()
    request = values.pop("request", None) or fakes.example_request()
    with session_scope(sessions) as db:
        store.ensure_owner(db, owner)
        row = store.create_analysis(
            db,
            owner_key=owner,
            request=request,
            fingerprint=values.pop("fingerprint", "req_test"),
            idempotency_key=None,
        )
        for key, value in values.items():
            setattr(row, key, value)
        return row.id


def lease_values(owner: str = "other-worker", seconds: int = 60) -> dict[str, Any]:
    return {"lease_owner": owner, "lease_expires_at": utcnow() + timedelta(seconds=seconds)}


def get(sessions: sessionmaker[Session], analysis_id: str) -> Analysis | None:
    with session_scope(sessions) as db:
        return db.get(Analysis, analysis_id)


def event_types(sessions: sessionmaker[Session], analysis_id: str) -> list[tuple[str, str]]:
    with session_scope(sessions) as db:
        return [(e.type.value, e.status.value) for e in store.events_after(db, analysis_id, 0)]
