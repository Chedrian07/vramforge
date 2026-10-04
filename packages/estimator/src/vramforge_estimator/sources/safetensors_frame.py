"""Safetensors header framing: 8-byte little-endian length, then a JSON header (plan.md §6.2).

Only the prefix and the header are read; tensor payloads are never touched. The JSON itself is
parsed and validated in ``inspection.safetensors_header``.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

# The safetensors library refuses headers above 100 MB (safetensors MAX_HEADER_SIZE).
SAFETENSORS_MAX_HEADER_BYTES = 100_000_000


class HeaderFrameError(ValueError):
    """The file does not start with a readable safetensors header."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class HeaderFrame:
    header_length: int
    header: bytes  # JSON bytes as stored (may end with space padding)
    file_size: int
    digest: str  # sha256 over the 8-byte prefix + header bytes

    @property
    def payload_size(self) -> int:
        return self.file_size - 8 - self.header_length


def read_header_frame(path: Path, *, max_header_bytes: int) -> HeaderFrame:
    """Read the header frame of a local ``.safetensors`` file, capped at `max_header_bytes`."""
    cap = min(max_header_bytes, SAFETENSORS_MAX_HEADER_BYTES)
    with path.open("rb") as handle:
        file_size = os.fstat(handle.fileno()).st_size
        prefix = handle.read(8)
        if len(prefix) < 8:
            raise HeaderFrameError("file_too_small")
        length = int.from_bytes(prefix, "little")
        if length < 2:
            raise HeaderFrameError("empty_header")
        if length > cap:
            raise HeaderFrameError("header_too_large")
        if 8 + length > file_size:
            raise HeaderFrameError("truncated_header")
        header = handle.read(length)
    if len(header) != length:
        raise HeaderFrameError("truncated_header")
    if not header.startswith(b"{"):
        raise HeaderFrameError("not_a_json_object")
    digest = hashlib.sha256(prefix + header).hexdigest()
    return HeaderFrame(header_length=length, header=header, file_size=file_size, digest=digest)
