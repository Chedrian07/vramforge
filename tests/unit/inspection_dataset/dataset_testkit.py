"""Helpers for the dataset inspection tests (imported by test modules, not collected)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from vramforge_estimator.schemas import FileEntry, SourceManifest, SourceType
from vramforge_estimator.sources import ResolvedSource

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "datasets"


def file_entries(path: Path) -> list[FileEntry]:
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    return [
        FileEntry(
            path=p.name if path.is_file() else p.relative_to(path).as_posix(),
            size=p.stat().st_size,
            sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
        )
        for p in files
    ]


def local_source(path: Path, entries: list[FileEntry] | None = None) -> ResolvedSource:
    entries = file_entries(path) if entries is None else entries
    digest = hashlib.sha256("".join(e.sha256 or "" for e in entries).encode()).hexdigest()
    manifest = SourceManifest(
        kind="dataset",
        source_type=SourceType.LOCAL,
        reference=f"local:fixtures/{path.name}",
        resolved_revision=f"sha256:{digest}",
        files=entries,
        fingerprint=digest,
    )
    return ResolvedSource(kind="dataset", manifest=manifest, local_path=path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def hf_source(
    repo_dir: Path, repo_id: str = "acme/demo-set", revision: str = "a" * 40
) -> ResolvedSource:
    """A pinned Hub dataset whose manifest lists `repo_dir` (served by the `fake_hub` fixture)."""
    manifest = SourceManifest(
        kind="dataset",
        source_type=SourceType.HUGGINGFACE,
        reference=f"hf:{repo_id}",
        repo_id=repo_id,
        resolved_revision=revision,
        files=file_entries(repo_dir),
        fingerprint=revision,
    )
    return ResolvedSource(kind="dataset", manifest=manifest, repo_id=repo_id, revision=revision)
