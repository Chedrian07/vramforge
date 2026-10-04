"""Write real safetensors files (8-byte LE length + JSON header) with an all-zero payload.

Used by tests to build synthetic local model directories. Payloads are created with
``truncate`` so large shapes (e.g. the MiMo shards) become sparse files that use no disk space.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ITEMSIZE = {
    "F64": 8,
    "F32": 4,
    "BF16": 2,
    "F16": 2,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}


def header_for(
    tensors: Mapping[str, tuple[str, Sequence[int]]], metadata: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """A header mapping with contiguous data_offsets in the given order."""
    header: dict[str, Any] = {}
    if metadata:
        header["__metadata__"] = dict(metadata)
    offset = 0
    for name, (dtype, shape) in tensors.items():
        size = math.prod(shape) * ITEMSIZE[dtype]
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    return header


def write_header(path: Path, header: Mapping[str, Any], *, raw: bytes | None = None) -> int:
    """Write `header` (or `raw` header bytes) followed by a sparse zero payload; returns file size."""
    if raw is None:
        raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
        raw += b" " * (-len(raw) % 8)  # same 8-byte alignment padding as the safetensors writer
    payload = max(
        (int(v["data_offsets"][1]) for k, v in header.items() if k != "__metadata__"), default=0
    )
    total = 8 + len(raw) + payload
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(len(raw).to_bytes(8, "little"))
        handle.write(raw)
        handle.truncate(total)
    return total


def write_safetensors(
    path: Path,
    tensors: Mapping[str, tuple[str, Sequence[int]]],
    metadata: Mapping[str, str] | None = None,
) -> int:
    return write_header(path, header_for(tensors, metadata))
