"""Training-node RAM, disk and analysis RAM estimates (plan §10.1)."""

from __future__ import annotations

from vf_fakes import H, V, make_cfg, make_inventory

from vramforge_estimator.memory import estimate_analysis_ram, estimate_disk, estimate_host_ram
from vramforge_estimator.schemas import (
    ConfigResolution,
    FileEntry,
    Objective,
    ReferenceStrategy,
    SourceManifest,
    SourceType,
    Strategy,
)

INV = make_inventory()
EMBED_BF16 = V * H * 2  # largest tensor of the fake inventory


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


def test_fp32_load_may_convert_on_the_host() -> None:
    est = estimate_host_ram(INV, make_cfg(inventory=INV, load_dtype="float32"), None)
    assert item(est, "model_load.staging").bytes_high == EMBED_BF16 + V * H * 4


def test_dpo_precompute_fingerprint_doubles_the_largest_fp32_tensor() -> None:
    cfg = make_cfg(Objective.DPO, inventory=INV, reference=ReferenceStrategy.PRECOMPUTED_LOG_PROBS)
    est = estimate_host_ram(INV, cfg, None)
    assert item(est, "dpo.reference_precompute_fingerprint").bytes_high == 2 * V * H * 4
    assert est.bytes_low == 2 * V * H * 4


def test_disk_total_is_unknown_while_an_item_is_unknown() -> None:
    count = ConfigResolution(field="lora.trainable_params", resolved=1000, reason="t")
    cfg = make_cfg(inventory=INV, resolutions=[count])  # QLoRA: bf16 adapter, AdamW bf16 states
    est = estimate_disk(INV, cfg, manifest("model", [100, 200]), manifest("dataset", [50]), 7)
    assert item(est, "download.model").bytes_high == 300
    assert item(est, "download.dataset").bytes_high == 50
    assert item(est, "analysis.row_length_artifact").bytes_high == 7
    assert item(est, "checkpoint.per_save").bytes_high == 1000 * 2 + 2 * 1000 * 2
    assert item(est, "training_node.datasets_cache").bytes_high is None
    assert est.total_bytes is None


def test_disk_items_without_sizes_stay_unknown() -> None:
    est = estimate_disk(INV, make_cfg(inventory=INV), manifest("model", [100, None]), None, None)
    model = item(est, "download.model")
    assert (model.bytes_low, model.bytes_high) == (100, None)
    assert item(est, "download.dataset").bytes_low is None
    assert item(est, "checkpoint.per_save").bytes_low is None  # no trainable count resolved


def test_full_finetune_checkpoint_counts_every_parameter() -> None:
    cfg = make_cfg(Objective.SFT, Strategy.FULL, inventory=INV)
    params = sum(t.numel for t in INV.tensors)
    est = estimate_disk(INV, cfg, None, None, None)
    assert item(est, "checkpoint.per_save").bytes_high == params * 2 + 2 * params * 2


def test_analysis_ram() -> None:
    est = estimate_analysis_ram(1000, 10_000)
    assert est.bytes_low == 10_000 + 8 * 1000 and est.bytes_high is None
    partial = estimate_analysis_ram(None, 10_000)
    assert partial.bytes_low == 10_000
    assert item(partial, "length_statistics").bytes_low is None
    assert estimate_analysis_ram(None, None).bytes_low is None
