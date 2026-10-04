"""plan §19.1: GiB conversion and dtype helpers."""

import pytest

from vramforge_estimator.units import GiB, canonical_dtype, round_up, tensor_bytes, to_gib


def test_gib_conversion() -> None:
    assert to_gib(1_073_741_824) == 1.0
    assert GiB == 1_073_741_824


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [("BF16", "bfloat16"), ("torch.float16", "float16"), ("fp32", "float32"), ("U8", "uint8")],
)
def test_canonical_dtype(raw: str, canonical: str) -> None:
    assert canonical_dtype(raw) == canonical


def test_unknown_dtype_raises() -> None:
    with pytest.raises(ValueError, match="unknown dtype"):
        canonical_dtype("float7")


def test_tensor_bytes_rounds_packed_4bit_up() -> None:
    assert tensor_bytes(4096 * 4096, "bfloat16") == 33_554_432
    assert tensor_bytes(3, "nf4") == 2


def test_round_up() -> None:
    assert round_up(100, None) == 100
    assert round_up(100, 1) == 100
    assert round_up(100, 64) == 128
