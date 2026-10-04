"""ORM models (docs/architecture.md §7).

Owner ids are sha256 digests of the `vf_owner` cookie, so a database dump does not contain the
bearer value itself. Every analysis/upload/cache row carries its owner and every query filters by
it (plan.md §16.3, §18).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base, BigIntPK, JsonType, UTCDateTime, utcnow

OWNER_ID_LEN = 64
ANALYSIS_ID_LEN = 32


class Owner(Base):
    __tablename__ = "owners"

    id: Mapped[str] = mapped_column(String(OWNER_ID_LEN), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)


class Analysis(Base):
    __tablename__ = "analyses"
    __table_args__ = (
        UniqueConstraint("owner_id", "idempotency_key", name="uq_analyses_owner_idempotency"),
        Index("ix_analyses_owner_status", "owner_id", "status"),
        Index("ix_analyses_status_lease", "status", "lease_expires_at"),
    )

    id: Mapped[str] = mapped_column(String(ANALYSIS_ID_LEN), primary_key=True)
    owner_id: Mapped[str] = mapped_column(
        String(OWNER_ID_LEN), ForeignKey("owners.id", ondelete="CASCADE"), index=True
    )
    idempotency_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    fingerprint: Mapped[str] = mapped_column(String(128))
    request: Mapped[dict[str, Any]] = mapped_column(JsonType)
    status: Mapped[str] = mapped_column(String(32))
    progress: Mapped[dict[str, Any] | None] = mapped_column(JsonType, nullable=True)
    result: Mapped[dict[str, Any] | None] = mapped_column(JsonType, nullable=True)
    error: Mapped[dict[str, Any] | None] = mapped_column(JsonType, nullable=True)
    lease_owner: Mapped[str | None] = mapped_column(String(128), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    attempt: Mapped[int] = mapped_column(Integer, default=1)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    cancel_requested_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    preprocess_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    # Relative to the data directory: "artifacts/<owner>/<analysis_id>".
    artifact_dir: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)


class AnalysisEventRow(Base):
    """SSE events. The autoincrement id is the SSE event id (monotonic per analysis because
    inserts for one analysis are serialized by a row lock on the analysis)."""

    __tablename__ = "analysis_events"
    __table_args__ = (Index("ix_analysis_events_analysis_id_id", "analysis_id", "id"),)

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    analysis_id: Mapped[str] = mapped_column(
        String(ANALYSIS_ID_LEN), ForeignKey("analyses.id", ondelete="CASCADE")
    )
    type: Mapped[str] = mapped_column(String(32))
    payload: Mapped[dict[str, Any]] = mapped_column(JsonType)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)


class Upload(Base):
    __tablename__ = "uploads"

    id: Mapped[str] = mapped_column(String(ANALYSIS_ID_LEN), primary_key=True)
    owner_id: Mapped[str] = mapped_column(
        String(OWNER_ID_LEN), ForeignKey("owners.id", ondelete="CASCADE"), index=True
    )
    filename: Mapped[str] = mapped_column(String(255))
    format: Mapped[str] = mapped_column(String(16))
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    sha256: Mapped[str] = mapped_column(String(64))
    # Relative to the data directory: "uploads/<owner>/<upload_id>/<filename>".
    path: Mapped[str] = mapped_column(String(512))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(UTCDateTime(), index=True)


class ScanCacheEntry(Base):
    """Owner-scoped reuse of a complete scan (plan.md §16.3: never shared across owners)."""

    __tablename__ = "scan_cache"
    __table_args__ = (
        UniqueConstraint("owner_id", "preprocess_key", name="uq_scan_cache_owner_key"),
    )

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    owner_id: Mapped[str] = mapped_column(
        String(OWNER_ID_LEN), ForeignKey("owners.id", ondelete="CASCADE"), index=True
    )
    preprocess_key: Mapped[str] = mapped_column(String(128))
    # Relative to the data directory.
    artifact_path: Mapped[str] = mapped_column(String(512))
    summary: Mapped[dict[str, Any]] = mapped_column(JsonType)
    analysis_id: Mapped[str | None] = mapped_column(String(ANALYSIS_ID_LEN), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    last_used_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)


__all__ = ["Analysis", "AnalysisEventRow", "Owner", "ScanCacheEntry", "Upload"]
