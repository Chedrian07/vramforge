"""The Alembic migration builds exactly the ORM schema and is idempotent (plan §16)."""

from pathlib import Path

from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import inspect

from vramforge_api.db import make_engine
from vramforge_api.migrate import upgrade_head
from vramforge_api.models import Base


def test_upgrade_head_matches_models_and_is_idempotent(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'm.db'}"
    upgrade_head(url, attempts=1)
    upgrade_head(url, attempts=1)  # second run is a no-op

    engine = make_engine(url)
    try:
        tables = set(inspect(engine).get_table_names())
        assert {"owners", "analyses", "analysis_events", "uploads", "scan_cache"} <= tables
        with engine.connect() as conn:
            ctx = MigrationContext.configure(conn, opts={"compare_type": True})
            diff = compare_metadata(ctx, Base.metadata)
        assert diff == []
    finally:
        engine.dispose()
