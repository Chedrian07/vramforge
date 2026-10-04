"""Initial schema: owners, analyses, analysis_events, uploads, scan_cache.

Revision ID: 0001_initial
Revises:
Create Date: 2026-10-04
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

JSON_T = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
BIGPK_T = sa.BigInteger().with_variant(sa.Integer(), "sqlite")


def _ts(name: str, *, nullable: bool = True) -> sa.Column:
    return sa.Column(name, sa.DateTime(timezone=True), nullable=nullable)


def upgrade() -> None:
    op.create_table(
        "owners",
        sa.Column("id", sa.String(64), primary_key=True),
        _ts("created_at", nullable=False),
        _ts("last_seen_at", nullable=False),
    )
    op.create_table(
        "analyses",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column(
            "owner_id",
            sa.String(64),
            sa.ForeignKey("owners.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("idempotency_key", sa.String(128), nullable=True),
        sa.Column("fingerprint", sa.String(128), nullable=False),
        sa.Column("request", JSON_T, nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("progress", JSON_T, nullable=True),
        sa.Column("result", JSON_T, nullable=True),
        sa.Column("error", JSON_T, nullable=True),
        sa.Column("lease_owner", sa.String(128), nullable=True),
        _ts("lease_expires_at"),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("cancel_requested", sa.Boolean(), nullable=False),
        _ts("cancel_requested_at"),
        sa.Column("preprocess_key", sa.String(128), nullable=True),
        sa.Column("artifact_dir", sa.String(255), nullable=False),
        _ts("created_at", nullable=False),
        _ts("updated_at", nullable=False),
        _ts("started_at"),
        _ts("finished_at"),
        sa.UniqueConstraint("owner_id", "idempotency_key", name="uq_analyses_owner_idempotency"),
    )
    op.create_index("ix_analyses_owner_id", "analyses", ["owner_id"])
    op.create_index("ix_analyses_owner_status", "analyses", ["owner_id", "status"])
    op.create_index("ix_analyses_status_lease", "analyses", ["status", "lease_expires_at"])

    op.create_table(
        "analysis_events",
        sa.Column("id", BIGPK_T, primary_key=True, autoincrement=True),
        sa.Column(
            "analysis_id",
            sa.String(32),
            sa.ForeignKey("analyses.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("type", sa.String(32), nullable=False),
        sa.Column("payload", JSON_T, nullable=False),
        _ts("created_at", nullable=False),
    )
    op.create_index("ix_analysis_events_analysis_id_id", "analysis_events", ["analysis_id", "id"])

    op.create_table(
        "uploads",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column(
            "owner_id",
            sa.String(64),
            sa.ForeignKey("owners.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("format", sa.String(16), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("path", sa.String(512), nullable=False),
        _ts("created_at", nullable=False),
        _ts("expires_at", nullable=False),
    )
    op.create_index("ix_uploads_owner_id", "uploads", ["owner_id"])
    op.create_index("ix_uploads_expires_at", "uploads", ["expires_at"])

    op.create_table(
        "scan_cache",
        sa.Column("id", BIGPK_T, primary_key=True, autoincrement=True),
        sa.Column(
            "owner_id",
            sa.String(64),
            sa.ForeignKey("owners.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("preprocess_key", sa.String(128), nullable=False),
        sa.Column("artifact_path", sa.String(512), nullable=False),
        sa.Column("summary", JSON_T, nullable=False),
        sa.Column("analysis_id", sa.String(32), nullable=True),
        _ts("created_at", nullable=False),
        _ts("last_used_at", nullable=False),
        sa.UniqueConstraint("owner_id", "preprocess_key", name="uq_scan_cache_owner_key"),
    )
    op.create_index("ix_scan_cache_owner_id", "scan_cache", ["owner_id"])


def downgrade() -> None:
    op.drop_index("ix_scan_cache_owner_id", table_name="scan_cache")
    op.drop_table("scan_cache")
    op.drop_index("ix_uploads_expires_at", table_name="uploads")
    op.drop_index("ix_uploads_owner_id", table_name="uploads")
    op.drop_table("uploads")
    op.drop_index("ix_analysis_events_analysis_id_id", table_name="analysis_events")
    op.drop_table("analysis_events")
    op.drop_index("ix_analyses_status_lease", table_name="analyses")
    op.drop_index("ix_analyses_owner_status", table_name="analyses")
    op.drop_index("ix_analyses_owner_id", table_name="analyses")
    op.drop_table("analyses")
    op.drop_table("owners")
