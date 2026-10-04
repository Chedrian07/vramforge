"""Real Hub inspection of the plan's example model (VRAMFORGE_NETWORK_TESTS=1).

Headers come from ranged requests and only config/index/tokenizer/template files are downloaded
(into HF_HOME when set, else a temporary directory). The live inventory must equal the one built
from the committed fixture.
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import pytest

from vramforge_estimator.inspection import inspect_model, load_tokenizer
from vramforge_estimator.schemas import ModelComponent, ModelSourceRef
from vramforge_estimator.sources import ResolvedSource, SourceAccess, resolve_model

pytestmark = pytest.mark.network

REPO = "XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B"
REVISION = "2367e865d009c13ac81713a2878291d33ab28177"


@pytest.fixture(scope="module")
def access(tmp_path_factory: pytest.TempPathFactory) -> SourceAccess:
    home = os.environ.get("HF_HOME")
    return SourceAccess(hf_home=Path(home) if home else tmp_path_factory.mktemp("hf"))


@pytest.fixture(scope="module")
def source(access: SourceAccess) -> ResolvedSource:
    return resolve_model(ModelSourceRef(reference=REPO, revision=REVISION), access)


def test_live_inventory_equals_fixture_inventory(
    source: ResolvedSource,
    access: SourceAccess,
    mimo_local: tuple[ResolvedSource, SourceAccess],
) -> None:
    live = inspect_model(source, access)
    assert len(live.tensors) == 760
    assert live.params_total == 9_409_813_744
    params = {c.component: c.params for c in live.by_component}
    assert params == {ModelComponent.TEXT: 8_953_803_264, ModelComponent.VISION: 456_010_480}
    assert Counter(m.component for m in live.linear_modules) == {
        ModelComponent.TEXT: 248,
        ModelComponent.VISION: 110,
    }
    assert {c.layer_type: c.count for c in live.facts.layer_type_counts} == {
        "full_attention": 8,
        "linear_attention": 24,
    }
    assert (live.facts.head_dim, live.facts.head_dim_source) == (256, "explicit")
    assert live.facts.tie_word_embeddings is False and live.facts.has_vision is True
    fixture = inspect_model(*mimo_local)
    assert live.inventory_hash == fixture.inventory_hash
    assert live == fixture


def test_live_tokenizer_manifest(source: ResolvedSource, access: SourceAccess) -> None:
    handle = load_tokenizer(source, access)
    m = handle.manifest
    assert m.tokenizer_class == "Qwen3_5Tokenizer"
    assert (m.vocab_size, m.config_vocab_size) == (248077, 248320)
    assert (m.bos_token, m.eos_token, m.pad_token) == (None, "<|im_end|>", "<|endoftext|>")
    assert (m.adds_bos_by_default, m.adds_eos_by_default) == (False, False)
    assert (m.model_max_length, m.model_max_length_is_sentinel) == (262144, False)
    assert m.chat_template_source == "chat_template.jinja"
    assert (
        m.chat_template_sha256 == "59a64ebb4df6d1489d09a91267cf3ceb106162d4a893c4f84833cfb8c897ff63"
    )
    assert m.has_generation_markers is True
    assert m.template_kwargs == ["enable_thinking"]
    assert m.template_parse_error is None
    # docs/research/example-model-dataset.md §1.1
    assert m.files_sha256["tokenizer.json"] == (
        "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
    )
    assert m.files_sha256["tokenizer_config.json"] == (
        "792fa3f0cb88b111e54ef3134c873531008c4df471d108da17903426e308aa7b"
    )
    assert set(m.files_sha256) == {
        "chat_template.jinja",
        "merges.txt",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    }
    ids = handle.tokenizer("hello world")["input_ids"]
    assert ids == [14556, 1814]
