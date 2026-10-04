"""Safetensors header framing and validation without reading payloads (plan.md §6.2, §18)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from vramforge_estimator.inspection.safetensors_header import (
    HeaderError,
    parse_header_bytes,
    validate_header,
)
from vramforge_estimator.sources.safetensors_frame import HeaderFrameError, read_header_frame


def _entry(dtype: str, shape: list[int], begin: int, end: int) -> dict[str, Any]:
    return {"dtype": dtype, "shape": shape, "data_offsets": [begin, end]}


def test_valid_header_with_metadata_and_scalar() -> None:
    header = {
        "__metadata__": {"format": "pt"},
        "b": _entry("F32", [2, 3], 0, 24),
        "a": _entry("BF16", [], 24, 26),
        "empty": _entry("F16", [0, 4], 26, 26),
    }
    tensors, metadata = validate_header(header, payload_size=26)
    assert metadata == {"format": "pt"}
    by_name = {t.name: t for t in tensors}
    assert (by_name["b"].dtype, by_name["b"].numel, by_name["b"].nbytes) == ("float32", 6, 24)
    assert (by_name["a"].shape, by_name["a"].numel) == ((), 1)
    assert by_name["empty"].numel == 0


def test_unsigned_complex_and_sub_byte_dtypes_have_exact_sizes() -> None:
    header = {
        "u16": _entry("U16", [3], 0, 6),
        "u32": _entry("U32", [3], 6, 18),
        "u64": _entry("U64", [3], 18, 42),
        "c64": _entry("C64", [3], 42, 66),
        "e8m0": _entry("F8_E8M0", [3], 66, 69),
        "f4": _entry("F4", [2, 3], 69, 72),  # 6 elements x 4 bits
        "f6a": _entry("F6_E2M3", [4], 72, 75),  # 4 elements x 6 bits
        "f6b": _entry("F6_E3M2", [8], 75, 81),
        "f4_empty": _entry("F4", [0, 3], 81, 81),
    }
    tensors, _ = validate_header(header, payload_size=81)
    assert {t.name: (t.dtype, t.numel, t.nbytes) for t in tensors} == {
        "u16": ("uint16", 3, 6),
        "u32": ("uint32", 3, 12),
        "u64": ("uint64", 3, 24),
        "c64": ("complex64", 3, 24),
        "e8m0": ("float8_e8m0fnu", 3, 3),
        "f4": ("float4_e2m1", 6, 3),
        "f6a": ("float6_e2m3", 4, 3),
        "f6b": ("float6_e3m2", 8, 6),
        "f4_empty": ("float4_e2m1", 0, 0),
    }


@pytest.mark.parametrize(
    ("header", "reason"),
    [
        ({"w": _entry("F32", [2], 0, 4)}, "size_mismatch"),
        ({"w": _entry("F4", [3], 0, 2)}, "misaligned_slice"),  # 12 bits, not whole bytes
        ({"w": _entry("F4", [], 0, 1)}, "misaligned_slice"),  # a scalar is 4 bits
        ({"w": _entry("F6_E3M2", [2], 0, 2)}, "misaligned_slice"),
        ({"w": _entry("F4", [4], 0, 4)}, "size_mismatch"),  # 2 bytes
        ({"w": _entry("F6_E2M3", [4], 0, 4)}, "size_mismatch"),  # 3 bytes
        ({"w": _entry("C64", [1], 0, 4)}, "size_mismatch"),  # 8 bytes
        ({"w": _entry("F32", [-1], 0, 4)}, "invalid_shape"),
        ({"w": _entry("F32", [True], 0, 4)}, "invalid_shape"),
        ({"w": _entry("F32", [1], 4, 0)}, "invalid_offsets"),
        ({"w": {"dtype": "F32", "shape": [1], "data_offsets": [0]}}, "invalid_offsets"),
        ({"w": _entry("F32", [1], 4, 8)}, "non_contiguous_offsets"),  # hole at the start
        ({"w": _entry("F32", [1], 0, 4), "v": _entry("F32", [1], 2, 6)}, "non_contiguous_offsets"),
        ({"w": _entry("Q3", [1], 0, 1)}, "unknown_dtype"),
        ({"w": _entry("Z9", [1], 0, 8)}, "unknown_dtype"),
        ({"w": [1, 2]}, "invalid_entry"),
        ({"__metadata__": {"a": 1}}, "invalid_metadata"),
    ],
)
def test_malformed_headers(header: dict[str, Any], reason: str) -> None:
    with pytest.raises(HeaderError) as exc:
        validate_header(header)
    assert exc.value.reason == reason


@pytest.mark.parametrize(
    ("dtype", "shape", "nbytes"),
    [
        ("F4", [4], 2),
        ("F4", [2, 3], 3),
        ("F4", [0], 0),
        ("F4", [3], 2),
        ("F4", [], 1),
        ("F4", [4], 4),
        ("F6_E2M3", [4], 3),
        ("F6_E3M2", [8], 6),
        ("F6_E3M2", [2], 2),
        ("F6_E2M3", [4], 4),
        ("F8_E8M0", [3], 3),
        ("U16", [3], 6),
        ("U32", [3], 12),
        ("U64", [3], 24),
        ("C64", [3], 24),
        ("C64", [3], 12),
    ],
)
def test_validation_agrees_with_the_safetensors_library(
    tmp_path: Path, dtype: str, shape: list[int], nbytes: int
) -> None:
    safetensors = pytest.importorskip("safetensors")
    header = {"w": _entry(dtype, shape, 0, nbytes)}
    path = _frame(tmp_path / "w.safetensors", json.dumps(header).encode(), nbytes)
    try:
        with safetensors.safe_open(str(path), framework="numpy") as handle:
            assert list(handle.keys()) == ["w"]
    except safetensors.SafetensorError:
        library_accepts = False
    else:
        library_accepts = True
    try:
        validate_header(header, payload_size=nbytes)
    except HeaderError:
        accepted = False
    else:
        accepted = True
    assert accepted is library_accepts


def test_payload_size_must_be_fully_indexed() -> None:
    header = {"w": _entry("F32", [1], 0, 4)}
    validate_header(header, payload_size=4)
    with pytest.raises(HeaderError) as exc:
        validate_header(header, payload_size=8)
    assert exc.value.reason == "payload_size_mismatch"


def test_parse_header_bytes() -> None:
    assert parse_header_bytes(b'{"a": {}}   ') == {"a": {}}
    for raw in (b"[1, 2]", b"\xff\xfe", b"{not json"):
        with pytest.raises(HeaderError):
            parse_header_bytes(raw)


def _frame(path: Path, header: bytes, payload: int = 0, declared: int | None = None) -> Path:
    with path.open("wb") as handle:
        handle.write((len(header) if declared is None else declared).to_bytes(8, "little"))
        handle.write(header)
        handle.truncate(8 + len(header) + payload)
    return path


def test_read_header_frame(tmp_path: Path) -> None:
    raw = json.dumps({"w": _entry("F32", [4], 0, 16)}).encode() + b"  "
    frame = read_header_frame(_frame(tmp_path / "a.safetensors", raw, 16), max_header_bytes=1 << 20)
    assert frame.header == raw
    assert frame.payload_size == 16
    assert len(frame.digest) == 64


@pytest.mark.parametrize(
    ("raw", "declared", "reason"),
    [
        (b"", 0, "empty_header"),
        (b"{}", 1 << 40, "header_too_large"),
        (b'{"a":', 4096, "truncated_header"),
        (b"[1, 2, 3]", None, "not_a_json_object"),
    ],
)
def test_read_header_frame_rejects(
    tmp_path: Path, raw: bytes, declared: int | None, reason: str
) -> None:
    path = _frame(tmp_path / "x.safetensors", raw, 0, declared)
    with pytest.raises(HeaderFrameError) as exc:
        read_header_frame(path, max_header_bytes=1 << 20)
    assert exc.value.reason == reason


def test_header_cap_is_applied_before_reading(tmp_path: Path) -> None:
    raw = b"{" + b" " * 4095 + b"}"
    path = _frame(tmp_path / "big.safetensors", raw)
    with pytest.raises(HeaderFrameError) as exc:
        read_header_frame(path, max_header_bytes=1024)
    assert exc.value.reason == "header_too_large"


def test_tiny_file(tmp_path: Path) -> None:
    path = tmp_path / "t.safetensors"
    path.write_bytes(b"\x01\x02")
    with pytest.raises(HeaderFrameError) as exc:
        read_header_frame(path, max_header_bytes=1024)
    assert exc.value.reason == "file_too_small"
