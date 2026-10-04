"""ORM models: SQLite-compatible JSON/timestamps and owner-scoped uniqueness."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy.exc import IntegrityError

from vramforge_api.db import Base, make_engine, session_factory, session_scope
from vramforge_api.models import Analysis, AnalysisEventRow, Owner


def _factory(tmp_path: Path):
    engine = make_engine(f"sqlite:///{tmp_path / 'models.db'}")
    Base.metadata.create_all(engine)
    return engine, session_factory(engine)


def _analysis(aid: str, owner: str, key: str | None) -> Analysis:
    return Analysis(
        id=aid,
        owner_id=owner,
        idempotency_key=key,
        fingerprint="req_x",
        request={"a": 1},
        status="QUEUED",
        attempt=1,
        cancel_requested=False,
        artifact_dir=f"artifacts/{owner}/{aid}",
    )


def test_timestamps_are_utc_aware_and_json_round_trips(tmp_path: Path) -> None:
    engine, factory = _factory(tmp_path)
    try:
        with session_scope(factory) as s:
            s.add(Owner(id="o" * 64))
            s.flush()  # no ORM relationships: parents are flushed explicitly
            s.add(_analysis("a" * 32, "o" * 64, None))
        with session_scope(factory) as s:
            row = s.get(Analysis, "a" * 32)
            assert row is not None
            assert row.created_at.tzinfo is not None
            assert row.created_at <= datetime.now(UTC)
            assert row.request == {"a": 1}
    finally:
        engine.dispose()


def test_idempotency_key_is_unique_per_owner_only(tmp_path: Path) -> None:
    engine, factory = _factory(tmp_path)
    try:
        with session_scope(factory) as s:
            s.add_all([Owner(id="o" * 64), Owner(id="p" * 64)])
            s.flush()
            s.add(_analysis("a" * 32, "o" * 64, "k1"))
            s.add(_analysis("b" * 32, "p" * 64, "k1"))  # same key, other owner: allowed
        with pytest.raises(IntegrityError), session_scope(factory) as s:
            s.add(_analysis("c" * 32, "o" * 64, "k1"))
    finally:
        engine.dispose()


def test_events_cascade_with_their_analysis(tmp_path: Path) -> None:
    engine, factory = _factory(tmp_path)
    try:
        with session_scope(factory) as s:
            s.add(Owner(id="o" * 64))
            s.flush()
            s.add(_analysis("a" * 32, "o" * 64, None))
        with session_scope(factory) as s:
            s.add(AnalysisEventRow(analysis_id="a" * 32, type="progress", payload={}))
            s.add(AnalysisEventRow(analysis_id="a" * 32, type="completed", payload={}))
        with session_scope(factory) as s:
            ids = [e.id for e in s.query(AnalysisEventRow).order_by(AnalysisEventRow.id)]
            assert ids == sorted(ids) and len(ids) == 2
            s.delete(s.get(Analysis, "a" * 32))
        with session_scope(factory) as s:
            assert s.query(AnalysisEventRow).count() == 0
    finally:
        engine.dispose()


def test_ensure_owner_survives_the_first_request_race(tmp_path: Path) -> None:
    """Two first requests of one browser both see no owner; the second insert must not fail."""
    from vramforge_api import store

    engine, factory = _factory(tmp_path)
    owner = "o" * 64
    try:
        with session_scope(factory) as first:
            store.ensure_owner(first, owner)  # the other request wins and commits
        with session_scope(factory) as second:
            real_get = second.get
            reads: list[object] = []

            def stale_get(model, key, **kwargs):  # type: ignore[no-untyped-def]
                if model is Owner and not reads:
                    reads.append(key)
                    return None  # read before the winner committed
                return real_get(model, key, **kwargs)

            second.get = stale_get  # type: ignore[method-assign]
            store.ensure_owner(second, owner)
            store.lock_owner(second, owner)  # the transaction is still usable
            second.add(_analysis("a" * 32, owner, None))
        with session_scope(factory) as check:
            assert check.query(Owner).count() == 1
            assert check.get(Analysis, "a" * 32) is not None
    finally:
        engine.dispose()


def test_ensure_owner_reraises_unrelated_integrity_errors(tmp_path: Path) -> None:
    from vramforge_api import store

    engine, factory = _factory(tmp_path)
    try:
        with pytest.raises(IntegrityError), session_scope(factory) as db:
            real_get = db.get

            def never_found(model, key, **kwargs):  # type: ignore[no-untyped-def]
                return None if model is Owner else real_get(model, key, **kwargs)

            db.get = never_found  # type: ignore[method-assign]
            db.add(Owner(id="o" * 64))
            db.flush()
            store.ensure_owner(db, "o" * 64)  # duplicate, but the row "cannot be found"
    finally:
        engine.dispose()
