"""Real Hugging Face Hub resolution of the plan's example model and dataset (plan.md §21).

Runs only with VRAMFORGE_NETWORK_TESTS=1. Metadata requests only; nothing is downloaded.
"""

from __future__ import annotations

import pytest

from vramforge_estimator.errors import EstimatorError
from vramforge_estimator.schemas import DatasetSourceRef, ErrorCode, ModelSourceRef, SourceType
from vramforge_estimator.sources import SourceAccess, resolve_dataset, resolve_model

pytestmark = pytest.mark.network

MODEL = "XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B"
MODEL_SHA = "2367e865d009c13ac81713a2878291d33ab28177"
DATASET = "CyberNative/Code_Vulnerability_Security_DPO"
DATASET_SHA = "81aeacf06cf43b16d7278a3a01f019a496a53c51"

# docs/research/example-model-dataset.md §1.1 and the Hub tree at MODEL_SHA
SHARDS = {
    "model-00001-of-00004.safetensors": (
        5276436216,
        "aab052180118aee34abc3029b54eaa49096aac606b97d703866b420dceb703c3",
    ),
    "model-00002-of-00004.safetensors": (
        5033171280,
        "7a0486565f06d25ac4628e9dba470dc3f604353471d240d5a0bf7128f64df396",
    ),
    "model-00003-of-00004.safetensors": (
        5234499504,
        "6c73207563d1879bfd6c143a028cc70be68edff56280458c71a43df4240300f4",
    ),
    "model-00004-of-00004.safetensors": (
        3275613848,
        "1379a7cf8c0b8555a39ab65a47e830e0eb45e776b46045734da3afd73c09eea2",
    ),
}


def test_example_model_resolves_to_pinned_commit() -> None:
    source = resolve_model(ModelSourceRef(reference=MODEL, revision=MODEL_SHA), SourceAccess())
    manifest = source.manifest
    assert manifest.source_type is SourceType.HUGGINGFACE
    assert manifest.reference == f"hf:{MODEL}"
    assert manifest.resolved_revision == MODEL_SHA
    assert (source.repo_id, source.revision) == (MODEL, MODEL_SHA)
    assert manifest.private is False and manifest.gated is False
    files = {f.path: f for f in manifest.files}
    for name, (size, lfs_sha) in SHARDS.items():
        assert (files[name].size, files[name].sha256) == (size, lfs_sha)
    assert files["config.json"].blob_id == "007c2eb8c73b9295f22491b1e4ab29d8b10f6e05"
    assert files["chat_template.jinja"].size == 3916
    assert files["tokenizer.json"].sha256 == (
        "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
    )


def test_example_model_url_with_tree_revision() -> None:
    url = f"https://huggingface.co/{MODEL}/tree/{MODEL_SHA}"
    manifest = resolve_model(ModelSourceRef(reference=url), SourceAccess()).manifest
    assert manifest.resolved_revision == MODEL_SHA
    assert manifest.requested_revision == MODEL_SHA


def test_example_dataset_resolves_to_pinned_commit() -> None:
    ref = DatasetSourceRef(
        reference=f"https://huggingface.co/datasets/{DATASET}/viewer/default/train?row=0",
        revision=DATASET_SHA,
    )
    source = resolve_dataset(ref, SourceAccess())
    manifest = source.manifest
    assert manifest.reference == f"hf-dataset:{DATASET}"
    assert manifest.resolved_revision == DATASET_SHA
    files = {f.path: f for f in manifest.files}
    data = files["secure_programming_dpo.json"]
    assert data.size == 6867898
    assert data.blob_id == "b2a8b2cae63af35f96dfc75440f1b50fac6a7346"
    assert any("row" in note for note in manifest.notes)


def test_missing_repository_and_revision() -> None:
    with pytest.raises(EstimatorError) as exc:
        resolve_model(
            ModelSourceRef(reference="XiaomiMiMo/does-not-exist-vramforge"), SourceAccess()
        )
    assert exc.value.issue.code is ErrorCode.SOURCE_NOT_FOUND
    with pytest.raises(EstimatorError) as exc:
        resolve_model(
            ModelSourceRef(reference=MODEL, revision="no-such-branch-vramforge"), SourceAccess()
        )
    assert exc.value.issue.code is ErrorCode.SOURCE_NOT_FOUND
    assert exc.value.issue.details["reason"] == "revision_not_found"
