"""Offline tokenizer fixtures for the preprocessing tests (tests/fixtures/golden/tokenizers)."""

from __future__ import annotations

from pathlib import Path

import pytest

from vramforge_estimator.inspection import TokenizerHandle
from vramforge_estimator.schemas import TokenizerManifest

TOKENIZERS = Path(__file__).parents[2] / "fixtures" / "golden" / "tokenizers"


def load_handle(name: str, *, chat_template: str | None = None) -> TokenizerHandle:
    from transformers import PreTrainedTokenizerFast

    tok = PreTrainedTokenizerFast.from_pretrained(TOKENIZERS / name)
    if chat_template is not None:
        tok.chat_template = chat_template
    template = tok.chat_template or ""
    manifest = TokenizerManifest(
        tokenizer_class=type(tok).__name__,
        vocab_size=len(tok),
        bos_token=tok.bos_token,
        eos_token=tok.eos_token,
        pad_token=tok.pad_token,
        chat_template_present=bool(template),
        has_generation_markers="generation -%}" in template or "{% generation %}" in template,
        fingerprint=f"fixture:{name}",
    )
    return TokenizerHandle(tokenizer=tok, manifest=manifest)


@pytest.fixture(scope="session")
def mimo() -> TokenizerHandle:
    return load_handle("mimo_bytelevel")


@pytest.fixture(scope="session")
def bos() -> TokenizerHandle:
    return load_handle("bos_bytelevel")


@pytest.fixture(scope="session")
def plain() -> TokenizerHandle:
    return load_handle("plain_bytelevel")


@pytest.fixture(scope="session")
def make_handle():
    """Factory for a fixture tokenizer with an optional chat-template override."""
    return load_handle
