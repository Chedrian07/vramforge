"""Training-node RAM, disk and analysis-server RAM (plan.md §10.1; docs/methodology.md#host-ram).

The same honesty rule as for GPU memory applies: a part without a basis is `None` with a reason,
never 0. RAM estimates therefore report `bytes_low` as the known lower bound (largest phase of
the modelled items) and `bytes_high = None` while the runtime baseline (Python, PyTorch/CUDA
libraries, tokenizer internals) has no measured basis. The disk total is `None` whenever an item
is unknown; the known items are still listed.
"""

from __future__ import annotations

from vramforge_estimator.architectures import get_adapter
from vramforge_estimator.errors import EstimatorError
from vramforge_estimator.schemas import (
    AnalysisRamEstimate,
    DiskEstimate,
    EstimateItem,
    Evidence,
    HostRamEstimate,
    ModelComponent,
    ModelInventory,
    ReferenceStrategy,
    ResolvedConfig,
    SourceManifest,
    Strategy,
    TensorInfo,
)
from vramforge_estimator.trainers.common import BNB_QMAP_BYTES, eight_bit_state_bytes
from vramforge_estimator.trainers.trainable import TrainableSlice, executed_slices
from vramforge_estimator.units import tensor_bytes

RUNTIME_BASELINE_NOTE = (
    "Python·PyTorch·CUDA 라이브러리의 기본 RSS는 측정 근거가 없어 상한을 정하지 않았습니다 "
    "(GPU 보정 M5에서 측정)."
)


def _in_scope(inventory: ModelInventory, cfg: ResolvedConfig) -> list[TensorInfo]:
    keep_all = cfg.loading_scope == "full_checkpoint"
    return [
        t
        for t in inventory.tensors
        if t.component is not ModelComponent.MTP
        and (keep_all or t.component is ModelComponent.TEXT)
    ]


def _baseline() -> EstimateItem:
    return EstimateItem(
        name="runtime_baseline",
        bytes_low=None,
        bytes_high=None,
        evidence=Evidence.UNKNOWN,
        note=RUNTIME_BASELINE_NOTE,
    )


def estimate_host_ram(
    inventory: ModelInventory, cfg: ResolvedConfig, model_source: SourceManifest | None
) -> HostRamEstimate:
    items: list[EstimateItem] = []
    phase_lows: list[int] = []
    tensors = _in_scope(inventory, cfg)
    if tensors:
        largest = max(tensors, key=lambda t: t.nbytes)
        converted = tensor_bytes(largest.numel, cfg.load_dtype)
        extra = 0 if largest.dtype == cfg.load_dtype else converted
        items.append(
            EstimateItem(
                name="model_load.staging",
                bytes_low=largest.nbytes,
                bytes_high=largest.nbytes + extra,
                evidence=Evidence.ANALYTIC,
                note=(
                    "가중치는 tensor 하나씩 CPU로 읽어 GPU로 옮깁니다 (가장 큰 tensor 기준). "
                    "load dtype 변환이 host에서 일어나면 변환본이 추가됩니다 (소스 기반 추론, "
                    "docs/research/loading-quantization-peft.md §4.3)."
                ),
            )
        )
        phase_lows.append(largest.nbytes)
        if cfg.dpo is not None and cfg.dpo.reference_strategy is (
            ReferenceStrategy.PRECOMPUTED_LOG_PROBS
        ):
            fingerprint = 2 * tensor_bytes(largest.numel, "float32")
            items.append(
                EstimateItem(
                    name="dpo.reference_precompute_fingerprint",
                    bytes_low=fingerprint,
                    bytes_high=fingerprint,
                    evidence=Evidence.ANALYTIC,
                    note=(
                        "precompute cache key를 만드는 hash_module이 가장 큰 tensor를 "
                        "fp32로 바꾸고 bytes 사본을 하나 더 만듭니다 (CPU 실측 2.00배, "
                        "docs/research/trl-sft-dpo.md §8.5)."
                    ),
                )
            )
            phase_lows.append(fingerprint)
    items.append(_baseline())
    _ = model_source  # mmap'd checkpoint pages are reclaimable page cache, not counted
    return HostRamEstimate(
        bytes_low=max(phase_lows) if phase_lows else None,
        bytes_high=None,
        items=items,
    )


def _files_item(name: str, source: SourceManifest | None, what: str) -> EstimateItem:
    if source is None or not source.files:
        return EstimateItem(
            name=name,
            bytes_low=None,
            bytes_high=None,
            evidence=Evidence.UNKNOWN,
            note=f"{what} 파일 목록이 없어 크기를 알 수 없습니다.",
        )
    sizes = [f.size for f in source.files]
    known = sum(s for s in sizes if s is not None)
    if any(s is None for s in sizes):
        return EstimateItem(
            name=name,
            bytes_low=known,
            bytes_high=None,
            evidence=Evidence.ANALYTIC,
            note=f"{what} 파일 일부의 크기를 알 수 없습니다.",
        )
    return EstimateItem(
        name=name,
        bytes_low=known,
        bytes_high=known,
        evidence=Evidence.ANALYTIC,
        note=f"{what} 원본 파일 합계 (revision 고정 snapshot).",
    )


def _optimizer_file_bytes(s: TrainableSlice, cfg: ResolvedConfig) -> tuple[int, int]:
    """Saved optimizer state of the executed part of a trainable bucket: AdamW keeps 2 states in
    the param dtype plus a 0-dim fp32 `step` per tensor; bnb 8-bit uses its per-tensor rule
    (docs/research/loading-quantization-peft.md §Q8.2-8.3)."""
    opt = cfg.optimizer
    if opt.eight_bit:
        return eight_bit_state_bytes(s, opt.min_8bit_size or 4096, opt.block_size or 256)
    states = opt.states_per_param * tensor_bytes(s.executed_numel, s.dtype)
    steps = 4 * s.executed_tensors
    return states + steps, states + steps


def _unknown_checkpoint(reason: str) -> EstimateItem:
    return EstimateItem(
        name="checkpoint.per_save",
        bytes_low=None,
        bytes_high=None,
        evidence=Evidence.UNKNOWN,
        note=reason,
    )


def _checkpoint_item(inventory: ModelInventory, cfg: ResolvedConfig) -> EstimateItem:
    """One save: the trained weights (PEFT: adapter file with LoRA, modules_to_save copies and
    trained biases; full FT: the whole model) plus the optimizer state, which exists only for
    parameters that receive gradients (text-only data never creates vision-tower state)."""
    if cfg.architecture_adapter is None:
        return _unknown_checkpoint("architecture adapter가 없어 checkpoint 크기를 알 수 없습니다.")
    try:
        arch = get_adapter(cfg.architecture_adapter)
        slices = executed_slices(arch.trainable_groups(inventory, cfg), inventory, cfg)
        if cfg.strategy is Strategy.FULL:
            resident = arch.resident_weights(inventory, cfg, [])
            sizes = [w.bytes_high for w in resident]
            weights: int | None = (
                None if any(v is None for v in sizes) else sum(v or 0 for v in sizes)
            )
        else:
            weights = sum(tensor_bytes(sl.numel, sl.dtype) for sl in slices)
    except EstimatorError:
        return _unknown_checkpoint(
            "학습 파라미터 구성을 알 수 없어 checkpoint 크기를 계산할 수 없습니다."
        )
    if weights is None:
        return _unknown_checkpoint(
            "모델 가중치 크기를 알 수 없어 checkpoint 크기를 계산할 수 없습니다."
        )
    state = [_optimizer_file_bytes(sl, cfg) for sl in slices if sl.executed_numel]
    low = weights + sum(lo for lo, _ in state)
    high = weights + sum(hi for _, hi in state)
    if cfg.optimizer.eight_bit and state:
        low, high = low + BNB_QMAP_BYTES, high + BNB_QMAP_BYTES
    return EstimateItem(
        name="checkpoint.per_save",
        bytes_low=low,
        bytes_high=high,
        evidence=Evidence.ANALYTIC,
        note=(
            "학습 가중치(PEFT: LoRA·modules_to_save 사본·학습 bias를 담은 adapter 파일, full: 모델 "
            "전체)와 gradient를 받는 파라미터의 optimizer state 1회 저장분입니다. scheduler·RNG·"
            "tokenizer 같은 작은 파일과 직렬화 header는 제외했고, 보존 개수(save_total_limit)만큼 "
            "배수가 됩니다."
        ),
    )


def estimate_disk(
    inventory: ModelInventory,
    cfg: ResolvedConfig,
    model_source: SourceManifest | None,
    dataset_source: SourceManifest | None,
    artifact_bytes: int | None,
) -> DiskEstimate:
    items = [
        _files_item("download.model", model_source, "모델"),
        _files_item("download.dataset", dataset_source, "데이터셋"),
        EstimateItem(
            name="analysis.row_length_artifact",
            bytes_low=artifact_bytes,
            bytes_high=artifact_bytes,
            evidence=Evidence.ANALYTIC if artifact_bytes is not None else Evidence.UNKNOWN,
            note=(
                "분석 서버의 row-length artifact (원문·token id 미저장)."
                if artifact_bytes is not None
                else "아직 artifact가 없어 크기를 알 수 없습니다."
            ),
        ),
        _checkpoint_item(inventory, cfg),
        EstimateItem(
            name="training_node.datasets_cache",
            bytes_low=None,
            bytes_high=None,
            evidence=Evidence.UNKNOWN,
            note=(
                "학습 노드에서 datasets가 만드는 Arrow·전처리(map) cache 크기는 "
                "조사되지 않았습니다."
            ),
        ),
    ]
    complete = all(i.bytes_high is not None for i in items)
    return DiskEstimate(
        items=items, total_bytes=sum(i.bytes_high or 0 for i in items) if complete else None
    )


def estimate_analysis_ram(rows: int | None, tokenizer_bytes: int | None) -> AnalysisRamEstimate:
    items = [
        EstimateItem(
            name="tokenizer",
            bytes_low=tokenizer_bytes,
            bytes_high=None,
            evidence=Evidence.ANALYTIC if tokenizer_bytes is not None else Evidence.UNKNOWN,
            note="tokenizer 파일은 로드할 때 메모리로 읽힙니다 (내부 자료구조 크기는 미측정).",
        ),
        EstimateItem(
            name="length_statistics",
            bytes_low=8 * rows if rows is not None else None,
            bytes_high=None,
            evidence=Evidence.ANALYTIC if rows is not None else Evidence.UNKNOWN,
            note="정확한 최대·분위수를 위해 row마다 길이 값(int64 이상)을 보관합니다.",
        ),
        _baseline(),
    ]
    lows = [i.bytes_low for i in items if i.bytes_low is not None]
    return AnalysisRamEstimate(bytes_low=sum(lows) if lows else None, bytes_high=None, items=items)


__all__ = ["estimate_analysis_ram", "estimate_disk", "estimate_host_ram"]
