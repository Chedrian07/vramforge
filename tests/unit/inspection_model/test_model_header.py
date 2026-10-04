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


@pytest.mark.parametrize(
    ("header", "reason"),
    [
        ({"w": _entry("F32", [2], 0, 4)}, "size_mismatch"),
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
