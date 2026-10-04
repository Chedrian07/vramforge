"""Resident weights and load-phase transients (plan.md §9.3).

Formulas: docs/research/loading-quantization-peft.md §3.2-3.5 (bitsandbytes 0.50.2 storage,
transformers 5.18.0 conversion targets, sync on-the-fly quantization). Anchors refer to
docs/methodology-architectures.md.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from vramforge_estimator.schemas import (
    AllocationCategory,
    AllocationSpec,
    Evidence,
    ModelComponent,
    ResolvedConfig,
    TensorInfo,
    TensorRole,
)
from vramforge_estimator.units import canonical_dtype, dtype_bytes, tensor_bytes

from .matching import bnb_skip_match, name_matches_any
from .structure import TOWER_COMPONENTS, ModelStructure

DOC = "methodology-architectures.md"
BNB_BLOCKSIZE = 64  # Params4bit default; transformers cannot change it (§3.1)
BNB_NESTED_BLOCKSIZE = 256  # double-quant blocksize, fixed (§3.1)
_CODE_BYTES = 16 * 4  # 4-bit code table, fp32, one per tensor
_NESTED_CODE_BYTES = 256 * 4  # nested code, fp32, copied per tensor
_OFFSET_BYTES = 4  # 0-dim fp32 absmax mean
# Non-quantized tensors load on ThreadPoolExecutor(max_workers=min(4, cpus)) (§Q4.7).
_ASYNC_LOAD_WORKERS = 4


@dataclass(frozen=True)
class Q4Bytes:
    payload: int  # packed nibbles
    metadata: int  # absmax (+ nested absmax, offset), code tables

    @property
    def total(self) -> int:
        return self.payload + self.metadata


def bnb_q4_bytes(
    numel: int,
    *,
    double_quant: bool,
    blocksize: int = BNB_BLOCKSIZE,
    nested_blocksize: int = BNB_NESTED_BLOCKSIZE,
) -> Q4Bytes:
    """Bytes of one bitsandbytes 4-bit weight (§3.3, verified byte-exact on CPU, E3/V2).

    B_q4_noDQ(n) = ceil(n/2) + 4*nb + 64
    B_q4_DQ(n)   = ceil(n/2) + nb + 4*ceil(nb/256) + 4 + 64 + 1024,   nb = ceil(n/64)
    """
    payload = math.ceil(numel / 2)
    nb = math.ceil(numel / blocksize)
    if double_quant:
        meta = nb + 4 * math.ceil(nb / nested_blocksize) + _OFFSET_BYTES
        meta += _CODE_BYTES + _NESTED_CODE_BYTES
    else:
        meta = 4 * nb + _CODE_BYTES
    return Q4Bytes(payload, meta)


def default_skip_modules(structure: ModelStructure) -> set[str]:
    """transformers default `modules_to_not_convert`: output embedding + tied modules
    (quantizers/base.py:38-62; the last-parameter module is lm_head for these classes)."""
    skip = {structure.output_embedding} if structure.output_embedding else set()
    for group in structure.inventory.tied_groups:
        skip.update(name.removesuffix(".weight") for name in group)
    return skip


def quantized_modules(structure: ModelStructure, cfg: ResolvedConfig) -> set[str]:
    """Linear modules converted to Linear4bit in the loading scope (§3.2).

    Only `nn.Linear` is converted; embeddings, norms, conv and raw parameters stay in the load
    dtype. The request schema has no skip-module knob, so the effective list is the transformers
    default (lm_head, tied modules) united with `quantization.skip_module_patterns`.
    """
    if not cfg.quantization.enabled:
        return set()
    skip = default_skip_modules(structure)
    patterns = list(cfg.quantization.skip_module_patterns)
    return {
        m.name
        for m in structure.linear_modules
        if structure.in_scope(m.component, cfg.loading_scope)
        and m.name not in skip
        and not bnb_skip_match(m.name, patterns)
    }


def _q4_cfg(cfg: ResolvedConfig) -> tuple[bool, int, int]:
    q = cfg.quantization
    return (
        q.double_quant,
        q.blocksize or BNB_BLOCKSIZE,
        q.nested_blocksize or BNB_NESTED_BLOCKSIZE,
    )


def resident_dtype(tensor: TensorInfo, cfg: ResolvedConfig) -> str:
    """Non-quantized tensors live in the load dtype (§3.1-3; Qwen3.5 has no
    _keep_in_fp32_modules); profile upcasts are honored."""
    if cfg.upcast_to_fp32_patterns and name_matches_any(tensor.module, cfg.upcast_to_fp32_patterns):
        return "float32"
    return canonical_dtype(cfg.load_dtype)


def _is_quantized_weight(tensor: TensorInfo, qmods: set[str]) -> bool:
    return (
        tensor.module in qmods
        and tensor.name.endswith(".weight")
        and tensor.role in (TensorRole.LINEAR_WEIGHT, TensorRole.LM_HEAD)
    )


@dataclass
class _WeightTally:
    payload: int = 0
    metadata: int = 0
    dense: int = 0
    q_tensors: int = 0
    dense_tensors: int = 0
    q_params: int = 0
    dense_params: int = 0


def _tally(tensors: Iterable[TensorInfo], qmods: set[str], cfg: ResolvedConfig) -> _WeightTally:
    dq, bs, nbs = _q4_cfg(cfg)
    tally = _WeightTally()
    for t in tensors:
        if _is_quantized_weight(t, qmods):
            q = bnb_q4_bytes(t.numel, double_quant=dq, blocksize=bs, nested_blocksize=nbs)
            tally.payload += q.payload
            tally.metadata += q.metadata
            tally.q_tensors += 1
            tally.q_params += t.numel
        else:
            tally.dense += tensor_bytes(t.numel, resident_dtype(t, cfg))
            tally.dense_tensors += 1
            tally.dense_params += t.numel
    return tally


def _alloc(
    name: str,
    nbytes: int,
    live_at: list[str],
    *,
    dtype: str | None,
    ref: str,
    dims: dict[str, int],
    note: str,
    shape: str,
) -> AllocationSpec:
    return AllocationSpec(
        name=name,
        category=AllocationCategory.WEIGHTS_BASE,
        shape_expression=shape,
        dims=dims,
        dtype=dtype,
        bytes_low=nbytes,
        bytes_high=nbytes,
        live_at=list(live_at),
        evidence=Evidence.ANALYTIC,
        formula_ref=f"{DOC}#{ref}",
        note=note,
    )


def resident_weight_allocations(
    structure: ModelStructure, cfg: ResolvedConfig, live_at: list[str]
) -> list[AllocationSpec]:
    tensors = structure.loaded_tensors(cfg.loading_scope)
    qmods = quantized_modules(structure, cfg)
    text = _tally((t for t in tensors if t.component not in TOWER_COMPONENTS), qmods, cfg)
    tower = _tally((t for t in tensors if t.component in TOWER_COMPONENTS), qmods, cfg)
    load = canonical_dtype(cfg.load_dtype)
    fmt = cfg.quantization.method or "bnb_nf4"
    qdtype = "fp4" if fmt == "bnb_fp4" else "nf4"
    dq = "DQ" if cfg.quantization.double_quant else "noDQ"
    allocs: list[AllocationSpec] = []
    if text.q_tensors:
        allocs.append(
            _alloc(
                "weights.base.q4_payload",
                text.payload,
                live_at,
                dtype=qdtype,
                ref="w-bnb-4bit",
                dims={"tensors": text.q_tensors, "params": text.q_params},
                note=f"텍스트 Linear {text.q_tensors}개의 4-bit packed payload (ceil(n/2)).",
                shape="Σ ceil(n/2)",
            )
        )
        allocs.append(
            _alloc(
                "weights.base.q4_metadata",
                text.metadata,
                live_at,
                dtype=None,
                ref="w-bnb-4bit",
                dims={"tensors": text.q_tensors, "blocksize": _q4_cfg(cfg)[1]},
                note=f"absmax·code 등 양자화 metadata ({dq}, blocksize 64).",
                shape=f"Σ (B_q4_{dq}(n) − ceil(n/2))",
            )
        )
    if text.dense_tensors:
        tied = " tied tensor는 한 번만 셉니다." if structure.tied_skip else ""
        allocs.append(
            _alloc(
                "weights.base.dense",
                text.dense,
                live_at,
                dtype=load,
                ref="w-dense",
                dims={"tensors": text.dense_tensors, "params": text.dense_params},
                note=(
                    "양자화되지 않은 텍스트 tensor(embedding, lm_head, norm, conv, bias 등)를 "
                    f"load dtype({load})으로 상주시킵니다.{tied}"
                ),
                shape="Σ numel × bytes(load_dtype)",
            )
        )
    if tower.q_tensors or tower.dense_tensors:
        allocs.append(
            _alloc(
                "weights.vision_tower",
                tower.payload + tower.metadata + tower.dense,
                live_at,
                dtype=None,
                ref="w-scope",
                dims={
                    "tensors": tower.q_tensors + tower.dense_tensors,
                    "params": tower.q_params + tower.dense_params,
                },
                note=(
                    f"loading scope '{cfg.loading_scope}'에서 로드되는 vision tower. "
                    f"4-bit payload {tower.payload:,} B + metadata {tower.metadata:,} B + "
                    f"비양자화 {tower.dense:,} B. 텍스트 전용 데이터에서는 실행되지 않습니다."
                ),
                shape="Σ B_q4(n) + Σ numel × bytes(load_dtype)",
            )
        )
    return allocs


def load_transient_allocations(
    structure: ModelStructure, cfg: ResolvedConfig, live_at: list[str]
) -> list[AllocationSpec]:
    """Extra device memory while loading (§3.5, §Q4).

    Quantized loading is synchronous, one tensor at a time: the largest quantized weight exists in
    the load dtype next to its outputs, plus the fp32 `_absmax` and `_absmax - offset` temporaries
    with double quant. If the checkpoint dtype differs from the load dtype and the conversion
    happens on the device (INFERRED, research open question 1) the source copy adds to it.
    """
    load = canonical_dtype(cfg.load_dtype)
    lb = dtype_bytes(load)
    tensors = structure.loaded_tensors(cfg.loading_scope)
    qmods = quantized_modules(structure, cfg)
    dq = cfg.quantization.double_quant
    blocksize = _q4_cfg(cfg)[1]

    def dq_temp(n: int) -> int:
        return 8 * math.ceil(n / blocksize) if dq else 0

    def conv(t: TensorInfo) -> int:
        return t.nbytes if canonical_dtype(t.dtype) != load else 0

    quant = [t for t in tensors if _is_quantized_weight(t, qmods)]
    if quant:
        largest = max(quant, key=lambda t: t.numel)
        low = math.ceil(largest.numel * lb) + dq_temp(largest.numel)
        high = low
        for t in tensors:
            if _is_quantized_weight(t, qmods):
                peak = math.ceil(t.numel * lb) + max(conv(t), dq_temp(t.numel))
            else:
                peak = conv(t)
            high = max(high, peak)
        note = (
            f"동기 on-the-fly 양자화: 가장 큰 양자화 weight({largest.module}) 1개가 load dtype "
            f"({load})으로 materialize된 순간"
            + (" + double-quant fp32 임시 absmax 2개" if dq else "")
            + "."
        )
        if high > low:
            note += " 상한은 checkpoint→load dtype 변환이 device에서 일어나는 경우(미검증)입니다."
        return [
            AllocationSpec(
                name="load.quantize_transient",
                category=AllocationCategory.LOAD_TRANSIENT,
                shape_expression="max_q [n_q·bytes(load) + (DQ ? 8·ceil(n_q/64) : 0)]",
                dims={"numel": largest.numel},
                dtype=load,
                bytes_low=low,
                bytes_high=high,
                live_at=list(live_at),
                evidence=Evidence.ANALYTIC if high == low else Evidence.ASSUMPTION,
                formula_ref=f"{DOC}#load-transient",
                note=note,
            )
        ]

    converted = sorted((conv(t) for t in tensors if conv(t)), reverse=True)
    if not converted:
        return []  # async load straight into the final dtype: no transient beyond the weights
    return [
        AllocationSpec(
            name="load.dtype_conversion_transient",
            category=AllocationCategory.LOAD_TRANSIENT,
            shape_expression="0 (host 변환) .. Σ top-4 checkpoint-dtype tensor (device 변환)",
            dims={"workers": _ASYNC_LOAD_WORKERS},
            dtype=None,
            bytes_low=0,
            bytes_high=sum(converted[:_ASYNC_LOAD_WORKERS]),
            live_at=list(live_at),
            evidence=Evidence.ASSUMPTION,
            formula_ref=f"{DOC}#load-transient",
            note=(
                "checkpoint dtype과 load dtype이 달라 변환이 필요합니다. host에서 변환하면 추가 "
                "GPU 메모리가 없고, device에서 변환하면 동시에 로드되는 최대 4개 tensor의 원본 "
                "사본이 더해집니다(미검증)."
            ),
        )
    ]


def device_map_load_bytes(structure: ModelStructure, cfg: ResolvedConfig) -> int:
    """`S_load` used by `caching_allocator_warmup` and the bnb `device_map="auto"` budget check:
    quantized weights at 0.5 B/param, others at the load dtype (§3.5, §Q4.4)."""
    load_b = dtype_bytes(cfg.load_dtype)
    qmods = quantized_modules(structure, cfg)
    total = 0.0
    for t in structure.loaded_tensors(cfg.loading_scope):
        total += t.numel * (0.5 if _is_quantized_weight(t, qmods) else load_b)
    return math.ceil(total)


def component_label(component: ModelComponent) -> str:
    return "text" if component not in TOWER_COMPONENTS else component.value
