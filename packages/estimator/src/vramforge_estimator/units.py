"""Byte units and dtype sizes.

All internal quantities are integer bytes (plan.md §0-9). GiB conversion is for display only and
must never be fed back into calculations.
"""

from __future__ import annotations

import math

KiB = 1024
MiB = 1024**2
GiB = 1024**3
GB = 1000**3

# Canonical dtype names follow torch naming.
DTYPE_BYTES: dict[str, float] = {
    "float64": 8,
    "float32": 4,
    "tfloat32": 4,
    "bfloat16": 2,
    "float16": 2,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
    "int64": 8,
    "int32": 4,
    "int16": 2,
    "int8": 1,
    "uint8": 1,
    "bool": 1,
    # Packed 4-bit payload: two elements per byte. Metadata is accounted separately.
    "nf4": 0.5,
    "fp4": 0.5,
}

# safetensors header dtype codes -> canonical names.
SAFETENSORS_DTYPES: dict[str, str] = {
    "F64": "float64",
    "F32": "float32",
    "BF16": "bfloat16",
    "F16": "float16",
    "F8_E4M3": "float8_e4m3fn",
    "F8_E5M2": "float8_e5m2",
    "I64": "int64",
    "I32": "int32",
    "I16": "int16",
    "I8": "int8",
    "U8": "uint8",
    "BOOL": "bool",
}


def canonical_dtype(name: str) -> str:
    """Normalize dtype spellings ("BF16", "torch.bfloat16", "bf16") to canonical names."""
    raw = name.strip()
    if raw in SAFETENSORS_DTYPES:
        return SAFETENSORS_DTYPES[raw]
    lowered = raw.lower().removeprefix("torch.")
    aliases = {
        "bf16": "bfloat16",
        "fp16": "float16",
        "half": "float16",
        "fp32": "float32",
        "float": "float32",
        "fp64": "float64",
        "double": "float64",
    }
    lowered = aliases.get(lowered, lowered)
    if lowered not in DTYPE_BYTES:
        raise ValueError(f"unknown dtype: {name!r}")
    return lowered


def dtype_bytes(name: str) -> float:
    return DTYPE_BYTES[canonical_dtype(name)]


def tensor_bytes(numel: int, dtype: str) -> int:
    """Bytes of a dense tensor; packed 4-bit dtypes round up to whole bytes."""
    return math.ceil(numel * dtype_bytes(dtype))


def to_gib(num_bytes: int) -> float:
    """Display-only conversion. Never use the result in further calculations."""
    return num_bytes / GiB


def to_gb(num_bytes: int) -> float:
    """Display-only decimal gigabytes (secondary notation)."""
    return num_bytes / GB


def round_up(value: int, multiple: int | None) -> int:
    """Round `value` up to a multiple (pad_to_multiple_of). `None`/1 means no rounding."""
    if not multiple or multiple <= 1:
        return value
    return ((value + multiple - 1) // multiple) * multiple
