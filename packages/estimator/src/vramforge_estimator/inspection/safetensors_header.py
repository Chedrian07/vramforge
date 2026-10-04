"""Validation of safetensors header mappings (plan.md §6.2).

Mirrors the checks the safetensors library applies before loading: known dtype, non-negative
shape, sub-byte tensors (FP4/FP6) ending on a byte boundary, ``data_offsets`` sized exactly
``numel × bits / 8`` and laid out contiguously from 0 without overlaps or holes (and, when the
file size is known, ending at the payload size).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from vramforge_estimator.units import SAFETENSORS_DTYPES, dtype_bytes


class HeaderError(ValueError):
    """Malformed or unsupported header; `reason` is a stable machine-readable code."""

    def __init__(self, reason: str, **details: object) -> None:
        super().__init__(reason)
        self.reason = reason
        self.details = details


@dataclass(frozen=True)
class HeaderTensor:
    name: str
    raw_dtype: str  # safetensors code, e.g. "BF16"
    dtype: str  # canonical name, e.g. "bfloat16"
    shape: tuple[int, ...]
    numel: int
    nbytes: int
    begin: int
    end: int


def parse_header_bytes(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise HeaderError("invalid_json") from None
    if not isinstance(value, dict):
        raise HeaderError("not_a_json_object")
    return value


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def validate_header(
    header: Mapping[str, Any], *, payload_size: int | None = None
) -> tuple[list[HeaderTensor], dict[str, str]]:
    """Tensors (in file order) and ``__metadata__`` of one shard."""
    metadata = header.get("__metadata__") or {}
    if not isinstance(metadata, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in metadata.items()
    ):
        raise HeaderError("invalid_metadata")

    tensors: list[HeaderTensor] = []
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        if not name or not isinstance(entry, dict):
            raise HeaderError("invalid_entry", tensor=name)
        raw_dtype = entry.get("dtype")
        shape = entry.get("shape")
        offsets = entry.get("data_offsets")
        if not isinstance(raw_dtype, str):
            raise HeaderError("invalid_entry", tensor=name)
        if not isinstance(shape, list) or not all(_is_int(d) for d in shape):
            raise HeaderError("invalid_shape", tensor=name)
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(_is_int(o) for o in offsets)
            or offsets[0] > offsets[1]
        ):
            raise HeaderError("invalid_offsets", tensor=name)
        dtype = SAFETENSORS_DTYPES.get(raw_dtype)
        if dtype is None:
            raise HeaderError("unknown_dtype", tensor=name, dtype=raw_dtype)
        numel = math.prod(shape)
        bits = numel * round(dtype_bytes(dtype) * 8)  # safetensors Dtype::bitsize (F4: 4, F6: 6)
        if bits % 8:
            # safetensors 0.8.0 refuses the file: "the slice does not end up at a byte boundary"
            raise HeaderError("misaligned_slice", tensor=name)
        nbytes = offsets[1] - offsets[0]
        if nbytes != bits // 8:
            raise HeaderError("size_mismatch", tensor=name)
        tensors.append(
            HeaderTensor(
                name=name,
                raw_dtype=raw_dtype,
                dtype=dtype,
                shape=tuple(shape),
                numel=numel,
                nbytes=nbytes,
                begin=offsets[0],
                end=offsets[1],
            )
        )

    expected = 0
    for tensor in sorted(tensors, key=lambda t: (t.begin, t.end)):
        if tensor.begin != expected:
            raise HeaderError("non_contiguous_offsets", tensor=tensor.name)
        expected = tensor.end
    if payload_size is not None and expected != payload_size:
        raise HeaderError("payload_size_mismatch")
    return tensors, dict(metadata)
