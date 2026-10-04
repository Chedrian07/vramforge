"""Training-node RAM, disk and analysis RAM estimates (plan §10.1)."""

from __future__ import annotations

import math

import pytest
from vf_fakes import (
    WEIGHTS,
    FakeArch,
    H,
    V,
    all_targets,
    lora_numel,
    make_cfg,
    make_inventory,
    text_targets,
)

from vramforge_estimator.memory import estimate_analysis_ram, estimate_disk, estimate_host_ram
from vramforge_estimator.memory import host as host_mod
from vramforge_estimator.schemas import (
    FileEntry,
    Objective,
    ReferenceStrategy,
    SourceManifest,
    SourceType,
    Strategy,
)

INV = make_inventory()
EMBED_BF16 = V * H * 2  # largest tensor of the fake inventory


@pytest.fixture
def fake_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(host_mod, "get_adapter", lambda adapter_id: FakeArch(adapter_id))


def manifest(kind: str, sizes: list[int | None]) -> SourceManifest:
    return SourceManifest(
        kind=kind,
        source_type=SourceType.HUGGINGFACE,
        reference=f"hf:org/{kind}",
        resolved_revision="0" * 40,
        files=[FileEntry(path=f"f{i}", size=s) for i, s in enumerate(sizes)],
        fingerprint="fp",
    )


def item(est, name):
    return next(i for i in est.items if i.name == name)


def test_host_ram_reports_a_lower_bound_without_runtime_baseline() -> None:
    est = estimate_host_ram(INV, make_cfg(inventory=INV), None)
    assert est.bytes_low == EMBED_BF16 and est.bytes_high is None
    staging = item(est, "model_load.staging")
    assert staging.bytes_low == staging.bytes_high == EMBED_BF16
    baseline = item(est, "runtime_baseline")
    assert baseline.bytes_low is None and baseline.note
    # plan §10.1: worker/prefetch/pinned-memory assumptions are stated, not sized as 0
    loader = item(est, "dataloader")
    assert loader.bytes_low is None and loader.bytes_high is None
    assert loader.note is not None and "dataloader_num_workers=0" in loader.note


def test_fp32_load_may_convert_on_the_host() -> None:
    est = estimate_host_ram(INV, make_cfg(inventory=INV, load_dtype="float32"), None)
    assert item(est, "model_load.staging").bytes_high == EMBED_BF16 + V * H * 4


def test_dpo_precompute_fingerprint_doubles_the_largest_fp32_tensor() -> None:
    cfg = make_cfg(Objective.DPO, inventory=INV, reference=ReferenceStrategy.PRECOMPUTED_LOG_PROBS)
    est = estimate_host_ram(INV, cfg, None)
    assert item(est, "dpo.reference_precompute_fingerprint").bytes_high == 2 * V * H * 4
    assert est.bytes_low == 2 * V * H * 4


def test_disk_total_is_unknown_while_an_item_is_unknown(fake_adapter) -> None:
    cfg = make_cfg(inventory=INV)  # QLoRA: bf16 adapter, AdamW (foreach) bf16 states
    est = estimate_disk(INV, cfg, manifest("model", [100, 200]), manifest("dataset", [50]), 7)
    assert item(est, "download.model").bytes_high == 300
    assert item(est, "download.dataset").bytes_high == 50
    assert item(est, "analysis.row_length_artifact").bytes_high == 7
    n = lora_numel(INV, text_targets(INV))
    tensors = 2 * len(text_targets(INV))  # A and B per module, one fp32 `step` each
    assert item(est, "checkpoint.per_save").bytes_high == 2 * n + 2 * 2 * n + 4 * tensors
    assert item(est, "training_node.datasets_cache").bytes_high is None
    assert est.total_bytes is None


def test_disk_items_without_sizes_stay_unknown() -> None:
    # "fake" is not a registered architecture adapter: the trainable state cannot be derived
    est = estimate_disk(INV, make_cfg(inventory=INV), manifest("model", [100, None]), None, None)
    model = item(est, "download.model")
    assert (model.bytes_low, model.bytes_high) == (100, None)
    assert item(est, "download.dataset").bytes_low is None
    checkpoint = item(est, "checkpoint.per_save")
    assert checkpoint.bytes_low is None and checkpoint.note


def test_peft_checkpoint_includes_modules_to_save_copies(fake_adapter) -> None:
    cfg = make_cfg(inventory=INV, modules_to_save=["lm_head"])
    n = lora_numel(INV, text_targets(INV))
    head = V * H  # bf16 trainable copy of lm_head (TRL casts trainable params of 4-bit models)
    tensors = 2 * len(text_targets(INV)) + 1
    expected = 2 * (n + head) + 2 * 2 * (n + head) + 4 * tensors
    assert item(estimate_disk(INV, cfg, None, None, None), "checkpoint.per_save").bytes_high == (
        expected
    )


def test_unexecuted_adapters_are_saved_without_optimizer_state(fake_adapter) -> None:
    inv = make_inventory(vision=True)
    cfg = make_cfg(inventory=inv, targets=all_targets(inv), optimizer="adamw_8bit")
    n_all, n_text = lora_numel(inv, all_targets(inv)), lora_numel(inv, text_targets(inv))
    by = {m.name: m for m in inv.linear_modules}
    state = 2 * 256 * 4  # shared qmaps
    for name in text_targets(inv):
        for n in (16 * by[name].in_features, by[name].out_features * 16):
            state += 8 * n if n < 4096 else 2 * n + 8 * math.ceil(n / 256)
    checkpoint = item(estimate_disk(inv, cfg, None, None, None), "checkpoint.per_save")
    assert checkpoint.bytes_low == checkpoint.bytes_high == 2 * n_all + state
    assert n_all > n_text


def test_full_finetune_checkpoint_saves_the_model_and_executed_state(fake_adapter) -> None:
    inv = make_inventory(vision=True)
    cfg = make_cfg(Objective.SFT, Strategy.FULL, inventory=inv)
    text = [t for t in inv.tensors if t.component.value == "text"]
    params = sum(t.numel for t in text)
    est = estimate_disk(inv, cfg, None, None, None)
    # weights: the whole resident model (FakeArch: WEIGHTS); state: text tensors only (bf16)
    expected = WEIGHTS + 2 * params * 2 + 4 * len(text)
    assert item(est, "checkpoint.per_save").bytes_high == expected


def test_analysis_ram() -> None:
    est = estimate_analysis_ram(1000, 10_000)
    assert est.bytes_low == 10_000 + 8 * 1000 and est.bytes_high is None
    partial = estimate_analysis_ram(None, 10_000)
    assert partial.bytes_low == 10_000
    assert item(partial, "length_statistics").bytes_low is None
    assert estimate_analysis_ram(None, None).bytes_low is None
