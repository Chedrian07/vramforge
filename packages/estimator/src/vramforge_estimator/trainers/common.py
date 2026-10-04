"""Shared schedule construction for the TRL 1.14.1 trainer adapters.

A trainer first lays out its timeline (all timepoints in temporal order), then attaches
allocations with explicit `live_at` lists. Rules that hold for every objective live here: model
loading (device-map budget check and load peak), resident weights, trainable state (adapter
weights, gradients, optimizer state), scope exclusions and the workspace assumptions.
Every formula is documented under the anchor given in `formula_ref` (docs/methodology.md).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from fractions import Fraction

from vramforge_estimator.architectures import ArchitectureAdapter
from vramforge_estimator.schemas import (
    AllocationCategory,
    AllocationSpec,
    Assumption,
    Evidence,
    ExcludedComponent,
    Issue,
    ModelInventory,
    Phase,
    ResolvedConfig,
    ScopeConfig,
    Strategy,
    Timepoint,
)
from vramforge_estimator.units import tensor_bytes

from .base import TrainingSchedule
from .ledger import alive_by_timepoint, contributions
from .trainable import TrainableSlice, executed_slices

METHODOLOGY = "methodology.md"
# transformers 5.18 device_map="auto" on one GPU without max_memory budgets 0.9 x free memory and
# the bitsandbytes 4-bit quantizer scales it by 0.90 again
# (docs/research/loading-quantization-peft.md §4.5, verified V9). Loading fails (4-bit) or offloads
# to CPU (dense) beyond this budget.
DEVICE_MAP_BUDGET = {"quantized": Fraction(81, 100), "dense": Fraction(9, 10)}
# The CUDA caching allocator rounds every allocation up to 512 B (torch 2.14.1
# c10/core/AllocatorConfig.h kMinBlockSize); fused AdamW keeps one fp32 `step` scalar per tensor
# on the device (torch/optim/adam.py `_init_group`).
MIN_BLOCK_BYTES = 512
# bitsandbytes 8-bit optimizers share two 256-entry fp32 quantization maps per optimizer.
BNB_QMAP_BYTES = 2 * 256 * 4


def ref(anchor: str) -> str:
    return f"{METHODOLOGY}#{anchor}"


def scaled(coefficient: str | int | Fraction, elements: int) -> int:
    """ceil(coefficient x elements) with exact decimal coefficients."""
    return math.ceil(Fraction(str(coefficient)) * elements)


def spec(
    name: str,
    category: AllocationCategory,
    low: int | None,
    high: int | None = None,
    *,
    live_at: Sequence[str],
    formula: str,
    evidence: Evidence = Evidence.ANALYTIC,
    note: str | None = None,
    shape: str | None = None,
    dims: dict[str, int] | None = None,
    dtype: str | None = None,
    saved: bool = False,
) -> AllocationSpec:
    """A ledger entry; `high` defaults to `low` (deterministic size)."""
    return AllocationSpec(
        name=name,
        category=category,
        shape_expression=shape,
        dims=dims or {},
        dtype=dtype,
        bytes_low=low,
        bytes_high=low if high is None else high,
        live_at=list(live_at),
        saved_for_backward=saved,
        evidence=evidence,
        formula_ref=ref(formula),
        note=note,
    )


def unknown(
    name: str, category: AllocationCategory, *, live_at: Sequence[str], formula: str, note: str
) -> AllocationSpec:
    return AllocationSpec(
        name=name,
        category=category,
        bytes_low=None,
        bytes_high=None,
        live_at=list(live_at),
        evidence=Evidence.UNKNOWN,
        formula_ref=ref(formula),
        note=note,
    )


# ---------------------------------------------------------------- builder


@dataclass
class ScheduleBuilder:
    timepoints: list[Timepoint] = field(default_factory=list)
    allocations: list[AllocationSpec] = field(default_factory=list)
    excluded: list[ExcludedComponent] = field(default_factory=list)
    assumptions: list[Assumption] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    # timepoints that are budget checks, not moments where memory is allocated
    budget_checks: set[str] = field(default_factory=set)

    def tp(self, phase: Phase, name: str, description: str) -> str:
        tp_id = f"{phase.value}:{name}"
        if any(t.id == tp_id for t in self.timepoints):
            raise ValueError(f"duplicate timepoint {tp_id}")
        self.timepoints.append(
            Timepoint(
                id=tp_id,
                phase=phase,
                name=name,
                description=description,
                order=len(self.timepoints),
            )
        )
        return tp_id

    def order(self, tp_id: str) -> int:
        return next(t.order for t in self.timepoints if t.id == tp_id)

    def ids(self, *phases: Phase) -> list[str]:
        return [t.id for t in self.timepoints if not phases or t.phase in phases]

    def from_(self, tp_id: str, *, checks: bool = True) -> list[str]:
        """Every timepoint at or after `tp_id` (optionally without budget checks)."""
        start = self.order(tp_id)
        return [
            t.id
            for t in self.timepoints
            if t.order >= start and (checks or t.id not in self.budget_checks)
        ]

    def step_ids(self) -> list[str]:
        """Timepoints of a steady-state training step (after loading and precompute)."""
        skip = {Phase.MODEL_LOAD_AND_QUANTIZE, Phase.REFERENCE_PRECOMPUTE}
        return [t.id for t in self.timepoints if t.phase not in skip]

    def add(self, *specs: AllocationSpec | Iterable[AllocationSpec]) -> None:
        for item in specs:
            if isinstance(item, AllocationSpec):
                self.allocations.append(item)
            else:
                self.allocations.extend(item)

    def assume(self, assumption_id: str, text: str, source: str | None = None) -> None:
        if all(a.id != assumption_id for a in self.assumptions):
            self.assumptions.append(Assumption(id=assumption_id, text=text, source=source))

    def build(self) -> TrainingSchedule:
        return TrainingSchedule(
            timepoints=list(self.timepoints),
            allocations=list(self.allocations),
            excluded=list(self.excluded),
            assumptions=list(self.assumptions),
            issues=list(self.issues),
        )


def extend_live_at(
    specs: Iterable[AllocationSpec], anchor: str, extra: Sequence[str]
) -> list[AllocationSpec]:
    """Allocations alive at `anchor` are also alive at `extra` (e.g. saved activations stay alive
    through the reference forward and the loss backward, which sit between the architecture
    adapter's forward/loss and backward timepoints)."""
    out = []
    for s in specs:
        if anchor in s.live_at and extra:
            missing = [e for e in extra if e not in s.live_at]
            s = s.model_copy(update={"live_at": [*s.live_at, *missing]})
        out.append(s)
    return out


def renamed(
    specs: Iterable[AllocationSpec], prefix: str, category: AllocationCategory | None = None
) -> list[AllocationSpec]:
    updates: dict[str, object] = {}
    if category is not None:
        updates["category"] = category
    return [s.model_copy(update={**updates, "name": f"{prefix}.{s.name}"}) for s in specs]


def residual_dtype(cfg: ResolvedConfig) -> str:
    """Residual stream / hidden-state dtype = load dtype (Linear4bit returns the input dtype and
    embeddings are not autocast; docs/research/loading-quantization-peft.md §Q9.3)."""
    return cfg.effective_dtypes.weights_nonquantized


def lm_head_trainable(cfg: ResolvedConfig) -> bool:
    if cfg.strategy is Strategy.FULL:
        return True
    return cfg.lora is not None and any(m.endswith("lm_head") for m in cfg.lora.modules_to_save)


def lm_head_param_dtype(cfg: ResolvedConfig) -> str:
    if cfg.strategy is Strategy.FULL:
        return cfg.effective_dtypes.gradient
    # modules_to_save copies keep the module dtype; TRL casts the trainable params of 4-bit models
    # to bf16 (docs/research/loading-quantization-peft.md §Q6.1)
    return "bfloat16" if cfg.quantization.enabled else residual_dtype(cfg)


# ---------------------------------------------------------------- model loading


@dataclass(frozen=True)
class LoadedModel:
    label: str
    check: str
    peak: str


def plan_model_load(b: ScheduleBuilder, label: str, title: str) -> LoadedModel:
    check = b.tp(
        Phase.MODEL_LOAD_AND_QUANTIZE,
        f"{label}_device_map_check",
        f"{title}: device_map='auto' 배치 전 가용 메모리 예산 검사",
    )
    b.budget_checks.add(check)
    peak = b.tp(
        Phase.MODEL_LOAD_AND_QUANTIZE,
        f"{label}_load_peak",
        f"{title}: 마지막(가장 큰) 가중치를 적재·양자화하는 순간",
    )
    return LoadedModel(label, check, peak)


def add_model_weights(
    b: ScheduleBuilder,
    arch: ArchitectureAdapter,
    inventory: ModelInventory,
    cfg: ResolvedConfig,
    load: LoadedModel,
    *,
    prefix: str | None = None,
    category: AllocationCategory | None = None,
) -> list[AllocationSpec]:
    """Resident weights from the load peak onwards, the load transient at the peak, and the
    device-map budget requirement at the check (docs/methodology.md#load-phase)."""
    live = b.from_(load.peak)
    weights = arch.resident_weights(inventory, cfg, live)
    transient = arch.load_transient(inventory, cfg, [load.peak])
    if prefix:
        weights = renamed(weights, prefix, category)
        transient = renamed(transient, prefix)
    b.add(weights, transient)
    kind = "quantized" if cfg.quantization.enabled else "dense"
    factor = DEVICE_MAP_BUDGET[kind]
    lows = [w.bytes_low for w in weights]
    highs = [w.bytes_high for w in weights]
    if any(v is None for v in [*lows, *highs]) or not weights:
        b.add(
            unknown(
                f"{load.label}.device_map_budget",
                AllocationCategory.LOAD_TRANSIENT,
                live_at=[load.check],
                formula="load-phase",
                note="가중치 크기를 몰라 device_map 예산 요구량을 산정할 수 없습니다.",
            )
        )
        return weights
    consequence = "부족하면 ValueError" if kind == "quantized" else "부족하면 CPU offload"
    low = math.ceil(sum(v or 0 for v in lows) / factor)
    high = math.ceil(sum(v or 0 for v in highs) / factor)
    b.add(
        spec(
            f"{load.label}.device_map_budget",
            AllocationCategory.LOAD_TRANSIENT,
            low,
            high,
            live_at=[load.check],
            formula="load-phase",
            note=(
                "실제 할당이 아니라 로딩 전 필요한 가용 메모리입니다: "
                f"가중치 ÷ {float(factor):g} ({consequence})."
            ),
        )
    )
    return weights


# ---------------------------------------------------------------- trainable state


def _eight_bit_bytes(s: TrainableSlice, min_size: int, block: int) -> tuple[int, int]:
    def per_tensor(n: int, embedding: bool) -> int:
        if embedding or n < min_size:
            return 8 * n
        return 2 * n + 8 * math.ceil(n / block)

    if s.tensors is not None:
        exact = sum(per_tensor(n, emb) for n, emb in s.tensors)
        return exact, exact
    n = s.executed_numel
    return 2 * n + 8 * math.ceil(n / block), 8 * n


def optimizer_state_bytes(s: TrainableSlice, cfg: ResolvedConfig) -> tuple[int, int, str]:
    """(low, high, note) of the optimizer state of the executed part of a trainable group
    (docs/research/loading-quantization-peft.md §Q8.2-8.3)."""
    opt = cfg.optimizer
    if opt.eight_bit:
        low, high = _eight_bit_bytes(s, opt.min_8bit_size or 4096, opt.block_size or 256)
        note = "bnb 8-bit: numel ≥ 4096이면 2 B + absmax, 작은 tensor와 nn.Embedding은 fp32 8 B."
        if opt.paged:
            note += " paged state도 VRAM에 포함했습니다 (paging 절감은 확정치로 쓰지 않음)."
        return low, high, note
    per = opt.states_per_param * tensor_bytes(s.executed_numel, s.group.dtype)
    if opt.fused:
        steps = s.executed_tensors * MIN_BLOCK_BYTES
        return (
            per + steps,
            per + steps,
            "AdamW: state 2개 × param dtype + tensor당 device step scalar.",
        )
    return per, per, "AdamW(foreach): state 2개 × param dtype, step scalar는 CPU."


def add_trainable_state(
    b: ScheduleBuilder,
    arch: ArchitectureAdapter,
    inventory: ModelInventory,
    cfg: ResolvedConfig,
    *,
    weights_live: Sequence[str],
    grads_live: Sequence[str],
    states_live: Sequence[str],
    step_tp: str,
) -> list[TrainableSlice]:
    """Adapter weights, gradients and optimizer state (docs/methodology.md#trainable-state)."""
    slices = executed_slices(arch.trainable_groups(inventory, cfg), inventory, cfg)
    qmap_added = False
    for s in slices:
        g = s.group
        if g.kind in ("lora", "modules_to_save"):
            b.add(
                spec(
                    f"adapter.{g.name}",
                    AllocationCategory.WEIGHTS_ADAPTER,
                    tensor_bytes(g.numel, g.dtype),
                    live_at=weights_live,
                    formula="trainable-state",
                    dims={"numel": g.numel, "tensors": g.tensor_count},
                    dtype=g.dtype,
                    note=g.note or None,
                )
            )
        if s.note:
            b.assume(f"trainable:{g.name}", s.note, "docs/research/loading-quantization-peft.md")
        if s.executed_numel == 0:
            continue
        b.add(
            spec(
                f"grad.{g.name}",
                AllocationCategory.GRADIENTS,
                tensor_bytes(s.executed_numel, g.dtype),
                live_at=grads_live,
                formula="gradients",
                dims={"numel": s.executed_numel},
                dtype=g.dtype,
                note=(
                    "grad dtype = param dtype; optimizer step 뒤 zero_grad(set_to_none=True)로 "
                    "해제됩니다."
                ),
            )
        )
        low, high, note = optimizer_state_bytes(s, cfg)
        if cfg.optimizer.eight_bit and not qmap_added:
            low, high, qmap_added = low + BNB_QMAP_BYTES, high + BNB_QMAP_BYTES, True
        b.add(
            spec(
                f"optimizer.{g.name}",
                AllocationCategory.OPTIMIZER_STATES,
                low,
                high,
                live_at=states_live,
                formula="optimizer-states",
                dims={"numel": s.executed_numel, "tensors": s.executed_tensors},
                dtype=cfg.effective_dtypes.optimizer_state,
                note=note,
            )
        )
        if not cfg.optimizer.fused and not cfg.optimizer.eight_bit:
            b.add(
                spec(
                    f"optimizer.{g.name}.foreach_sqrt",
                    AllocationCategory.WORKSPACE,
                    tensor_bytes(s.executed_numel, g.dtype),
                    live_at=[step_tp],
                    formula="optimizer-step",
                    dtype=g.dtype,
                    note=(
                        "foreach AdamW의 exp_avg_sq_sqrt 임시 목록 "
                        "(torch 2.14.1 torch/optim/adam.py _multi_tensor_adam)."
                    ),
                )
            )
    return slices


def grads_live(cfg: ResolvedConfig, fb_ids: list[str], backward: str, step_tp: str) -> list[str]:
    """Gradients exist from the first backward of the accumulation window until zero_grad after
    the optimizer step: with accumulation > 1 every micro-step after the first runs its forward
    with the full gradient set resident (docs/methodology.md#gradients)."""
    window = fb_ids if cfg.accumulation > 1 else [backward]
    return [*window, step_tp]


def lm_head_input_saved(
    b: ScheduleBuilder, cfg: ResolvedConfig, positions: int, hidden: int, live_at: list[str]
) -> None:
    """A trainable lm_head saves its input (the final-norm output) for its weight gradient; a
    frozen one does not (docs/research/architecture-memory.md §10.1 row 27)."""
    if not lm_head_trainable(cfg):
        return
    res = residual_dtype(cfg)
    b.add(
        spec(
            "lm_head.input",
            AllocationCategory.SAVED_ACTIVATIONS,
            tensor_bytes(positions * hidden, res),
            live_at=live_at,
            formula="lm-head-input",
            shape="positions × H",
            dims={"positions": positions, "hidden": hidden},
            dtype=res,
            saved=True,
            note="학습되는 lm_head가 weight gradient를 위해 입력(최종 norm 출력)을 저장합니다.",
        )
    )


# ---------------------------------------------------------------- scope, workspace


def add_scope_phases(b: ScheduleBuilder, scope: ScopeConfig) -> tuple[str | None, str | None]:
    """EVALUATION and CHECKPOINT_SAVE_OR_CONSOLIDATE: excluded unless requested; when requested
    their working set is unknown (no eval batch plan, save path not researched)."""
    eval_tp = save_tp = None
    if scope.include_evaluation:
        eval_tp = b.tp(Phase.EVALUATION, "forward", "평가 batch의 no-grad forward와 loss")
    else:
        b.excluded.append(
            ExcludedComponent(
                name=Phase.EVALUATION.value,
                reason="평가 단계는 범위에서 제외했습니다 (scope.include_evaluation=false).",
            )
        )
    if scope.include_checkpoint_save:
        save_tp = b.tp(Phase.CHECKPOINT_SAVE_OR_CONSOLIDATE, "save", "checkpoint 저장")
    else:
        b.excluded.append(
            ExcludedComponent(
                name=Phase.CHECKPOINT_SAVE_OR_CONSOLIDATE.value,
                reason=(
                    "checkpoint 저장 단계는 범위에서 제외했습니다 "
                    "(scope.include_checkpoint_save=false)."
                ),
            )
        )
    return eval_tp, save_tp


def add_scope_unknowns(b: ScheduleBuilder, eval_tp: str | None, save_tp: str | None) -> None:
    if eval_tp:
        b.add(
            unknown(
                "evaluation.working_set",
                AllocationCategory.SAVED_ACTIVATIONS,
                live_at=[eval_tp],
                formula="unknown-vs-excluded",
                note="평가 데이터의 batch 계획이 없어 평가 forward 메모리를 산정할 수 없습니다.",
            )
        )
    if save_tp:
        b.add(
            unknown(
                "checkpoint.save_transient",
                AllocationCategory.WORKSPACE,
                live_at=[save_tp],
                formula="unknown-vs-excluded",
                note="checkpoint 저장 경로의 GPU 임시 메모리는 조사되지 않았습니다.",
            )
        )


def add_workspace(b: ScheduleBuilder, cfg: ResolvedConfig) -> None:
    """CUDA context, library workspace and per-timepoint allocator slack as ASSUMPTION entries
    with the profile's low/high (docs/methodology.md#workspace-assumptions). Call last."""
    ws = cfg.workspace
    everywhere = b.ids()
    compute = [t for t in everywhere if not t.startswith(Phase.MODEL_LOAD_AND_QUANTIZE.value)]
    b.add(
        spec(
            "cuda_context",
            AllocationCategory.NON_FRAMEWORK,
            ws.cuda_context_bytes[0],
            ws.cuda_context_bytes[1],
            live_at=everywhere,
            formula="workspace-assumptions",
            evidence=Evidence.ASSUMPTION,
            note="CUDA context·모듈 로드(드라이버, cuBLAS handle 포함) 가정 범위.",
        ),
        spec(
            "library_workspace",
            AllocationCategory.WORKSPACE,
            ws.library_workspace_bytes[0],
            ws.library_workspace_bytes[1],
            live_at=compute,
            formula="workspace-assumptions",
            evidence=Evidence.ASSUMPTION,
            note="cuBLAS/cuBLASLt·cuDNN workspace 가정 범위 (첫 연산부터).",
        ),
    )
    frac_low, frac_high = (Fraction(str(v)) for v in ws.allocator_slack_fraction)
    alive = alive_by_timepoint(b.timepoints, b.allocations)
    slack: list[AllocationSpec] = []
    for tp_id in everywhere:
        if tp_id in b.budget_checks:
            continue
        parts = [
            c
            for c in contributions(alive[tp_id])
            if c.spec.category is not AllocationCategory.NON_FRAMEWORK
        ]
        if any(c.unknown for c in parts):
            continue  # the timepoint is unknown anyway; slack of an unknown total is not sized
        base_low = sum(c.bytes_low or 0 for c in parts)
        base_high = sum(c.bytes_high or 0 for c in parts)
        slack.append(
            spec(
                f"allocator_slack@{tp_id}",
                AllocationCategory.ALLOCATOR_SLACK,
                math.ceil(base_low * frac_low),
                math.ceil(base_high * frac_high),
                live_at=[tp_id],
                formula="workspace-assumptions",
                evidence=Evidence.ASSUMPTION,
                note=(
                    f"caching allocator 여유(reserved−allocated) 가정: 같은 시점 할당량의 "
                    f"{float(frac_low):.0%}–{float(frac_high):.0%}."
                ),
            )
        )
    b.add(slack)
    b.assume(
        "workspace",
        "CUDA context, 라이브러리 workspace, allocator 여유는 profile의 가정 범위이며 "
        "GPU 보정(M5) 전까지 실측값이 아닙니다.",
        "docs/research/loading-quantization-peft.md §Q10",
    )


def common_assumptions(b: ScheduleBuilder, cfg: ResolvedConfig) -> None:
    b.assume(
        "steady_state",
        "학습 step은 optimizer state가 이미 생성된 정상 상태(두 번째 step 이후)로 계산합니다.",
        "docs/research/loading-quantization-peft.md §3.3",
    )
    b.assume(
        "worst_microbatch",
        "gradient accumulation 창에서 가장 불리한 micro-step(이미 누적된 gradient가 있는 경우)을 "
        "기준으로 하며, accumulation 횟수로 activation을 곱하지 않습니다.",
        "plan.md §9.4",
    )
    b.assume(
        "single_gpu_native_amp",
        f"단일 GPU, accelerate native AMP(bf16 autocast, fp32 출력 변환), load dtype "
        f"{cfg.load_dtype}.",
        "docs/research/trl-sft-dpo.md §8.4",
    )


__all__ = [
    "DEVICE_MAP_BUDGET",
    "LoadedModel",
    "ScheduleBuilder",
    "add_model_weights",
    "add_scope_phases",
    "add_scope_unknowns",
    "add_trainable_state",
    "add_workspace",
    "common_assumptions",
    "extend_live_at",
    "grads_live",
    "lm_head_input_saved",
    "lm_head_param_dtype",
    "lm_head_trainable",
    "optimizer_state_bytes",
    "plan_model_load",
    "ref",
    "renamed",
    "residual_dtype",
    "scaled",
    "spec",
    "unknown",
]
