"""SourceResolver contract (plan.md §6.1, §16.2, §18).

Resolvers turn a user reference into an immutable `SourceManifest` plus a handle that later
stages use to read files. They never download model weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from vramforge_estimator.schemas import SourceManifest


@dataclass(frozen=True)
class SourceAccess:
    """Credentials, roots and limits available to resolvers (server-side only, never serialized)."""

    hf_token: str | None = None
    # name -> absolute container path of a read-only root, referenced as "local:<name>/<rel>".
    local_roots: dict[str, Path] = field(default_factory=dict)
    uploads_dir: Path | None = None
    hf_home: Path | None = None
    allow_private_network: bool = False
    http_timeout_s: float = 30.0
    max_metadata_bytes: int = 64 * 1024 * 1024  # cap for config/tokenizer/header downloads


@dataclass(frozen=True)
class ResolvedSource:
    kind: Literal["model", "dataset"]
    manifest: SourceManifest
    # Exactly one of the following identifies where files are read from.
    repo_id: str | None = None  # HF repo id; read at `revision` (immutable commit sha)
    revision: str | None = None
    local_path: Path | None = None  # container path for local roots and uploads
