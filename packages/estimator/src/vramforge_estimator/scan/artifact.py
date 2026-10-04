"""Row-length artifact: Parquet part files of `LengthRecord` columns (plan.md §7.6, §16.2).

Layout under ``<artifact_dir>/lengths/``: ``part-00000.parquet``, ... written atomically
(temp file + rename) and ``manifest.json`` once the scan ends. Every part's Parquet schema
metadata carries the artifact schema id and the ``preprocess_key`` of the scan that wrote it, so
a loaded table knows which preprocessing produced it (plan §16.3). Raw text and token ids are
never stored. pyarrow is imported lazily so the core package imports without the analysis extra.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from vramforge_estimator.schemas import LengthRecord

LENGTHS_DIR = "lengths"
MANIFEST = "manifest.json"
PART_GLOB = "part-*.parquet"
SCHEMA_ID = "vramforge/length-record@1"
SCHEMA_META = "vramforge.schema"  # Parquet schema metadata keys of every part
PREPROCESS_KEY_META = "vramforge.preprocess_key"

COLUMNS: tuple[str, ...] = tuple(LengthRecord.model_fields)
_INT_COLUMNS = frozenset(
    {
        "row_index",
        "prompt_tokens",
        "completion_tokens",
        "sequence_tokens",
        "loss_token_count",
        "chosen_total_tokens",
        "rejected_total_tokens",
        "chosen_completion_tokens",
        "rejected_completion_tokens",
    }
)


def _schema(metadata: dict[str, str]) -> Any:
    import pyarrow as pa

    fields = [pa.field(c, pa.int64() if c in _INT_COLUMNS else pa.string()) for c in COLUMNS]
    return pa.schema(fields, metadata={k.encode(): v.encode() for k, v in metadata.items()})


def part_name(seq: int) -> str:
    return f"part-{seq:05d}.parquet"


def write_part(
    directory: Path, seq: int, columns: dict[str, list[Any]], metadata: dict[str, str]
) -> str:
    """Write one part atomically and return its file name."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    directory.mkdir(parents=True, exist_ok=True)
    name = part_name(seq)
    table = pa.table({c: columns[c] for c in COLUMNS}, schema=_schema(metadata))
    tmp = directory / f".{name}.tmp"
    pq.write_table(table, tmp)
    os.replace(tmp, directory / name)
    return name


def write_manifest(directory: Path, manifest: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    tmp = directory / f".{MANIFEST}.tmp"
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    os.replace(tmp, directory / MANIFEST)


def read_manifest(directory: Path) -> dict[str, Any] | None:
    path = directory / MANIFEST
    if not path.is_file():
        return None
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def remove_parts(directory: Path, keep: set[str] | frozenset[str] = frozenset()) -> None:
    """Delete part/temp files of this artifact that are not in `keep` (stale or orphaned)."""
    if not directory.is_dir():
        return
    for path in [*directory.glob(PART_GLOB), *directory.glob(".*.tmp")]:
        if path.name not in keep:
            path.unlink(missing_ok=True)
    if not keep:
        (directory / MANIFEST).unlink(missing_ok=True)


def part_paths(artifact_path: Path) -> list[Path]:
    """Committed part files in order (manifest first, else every part on disk)."""
    if artifact_path.is_file():
        return [artifact_path]
    manifest = read_manifest(artifact_path)
    if manifest is not None:
        return [artifact_path / name for name in manifest["parts"]]
    return sorted(artifact_path.glob(PART_GLOB))


def iter_parts(
    paths: list[Path], columns: list[str]
) -> Iterator[tuple[dict[str, list[Any]], dict[str, str]]]:
    """Yield the requested columns of each part as Python lists, with the part's metadata."""
    import pyarrow.parquet as pq

    for path in paths:
        table = pq.read_table(path, columns=columns)
        raw = table.schema.metadata or {}
        metadata = {k.decode("utf-8"): v.decode("utf-8") for k, v in raw.items()}
        yield {c: table.column(c).to_pylist() for c in columns}, metadata


def iter_columns(paths: list[Path], columns: list[str]) -> Iterator[dict[str, list[Any]]]:
    """Yield the requested columns of each part as Python lists."""
    for cols, _metadata in iter_parts(paths, columns):
        yield cols


__all__ = [
    "COLUMNS",
    "LENGTHS_DIR",
    "PREPROCESS_KEY_META",
    "SCHEMA_ID",
    "SCHEMA_META",
    "iter_columns",
    "iter_parts",
    "part_name",
    "part_paths",
    "read_manifest",
    "remove_parts",
    "write_manifest",
    "write_part",
]
