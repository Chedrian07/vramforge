"""TRL 1.14.1 DPOTrainer schedule (docs/methodology.md#dpo).

The collator stacks chosen then rejected rows (2B rows, padded to the longest branch) and the
policy runs one forward over all positions; accelerate's native AMP wrapper returns fp32 logits,
which TRL's fused log-prob kernel saves for backward. The reference runs after the policy forward
while the policy graph and logits are alive, so policy and reference fp32 logits coexist until
`_compute_loss` returns (docs/research/trl-sft-dpo.md §8, implications D/E).

Reference strategies: `frozen_base_switch` (PEFT adapter disabled, no extra weights),
`standalone_model` (a second full model loaded in `__init__`), `precomputed_log_probs` (a no-grad
pass over the dataset in `__init__`; no reference forward during training).
"""

from __future__ import annotations

from vramforge_estimator.architectures import ArchitectureAdapter, StepTimepoints
from vramforge_estimator.schemas import (
    AllocationCategory,
    BatchPlan,
    BatchShape,
    ModelInventory,
    Objective,
    Phase,
    ReferenceStrategy,
    ResolvedConfig,
    ScopeConfig,
    SequenceShape,
)
from vramforge_estimator.units import tensor_bytes

from .base import TrainingSchedule
from .common import (
    POLICY_PREFIX,
    ScheduleBuilder,
    add_model_weights,
    add_scope_phases,
    add_scope_unknowns,
    add_trainable_state,
    add_workspace,
    common_assumptions,
    extend_live_at,
    grads_live,
    lm_head_input_saved,
    plan_model_load,
    spec,
    unknown,
)

LOSS = AllocationCategory.LOGITS_AND_LOSS


def add_dpo_logits(
    b: ScheduleBuilder,
    cfg: ResolvedConfig,
    hidden: int,
    vocab: int,
    shape: BatchShape,
    tps: list[str],
    ref_fwd: str | None,
    loss: str,
    loss_bwd: str,
) -> None:
    """Policy/reference logits and the fused-kernel buffers (docs/methodology.md#dpo-logits).
    `tps` = the policy timepoints from the forward through the loss backward."""
    rows = shape.sequences_per_forward  # 2 x pairs
    seq = shape.padded_length
    if cfg.loss_path != "trl_fused_logprob":
        b.add(
            unknown(
                "loss.unsupported_path",
                LOSS,
                live_at=[loss, loss_bwd],
                formula="dpo-logits",
                note=f"loss 경로 {cfg.loss_path!r}의 메모리 모델이 없습니다.",
            )
        )
        return
    positions = rows * seq
    nv = positions * vocab
    shifted = rows * max(seq - 1, 1)
    compute = cfg.effective_dtypes.compute
    b.add(
        spec(
            "logits.policy.lm_head_output",
            LOSS,
            tensor_bytes(nv, compute),
            live_at=[tps[0]],
            formula="dpo-logits",
            shape="2B × T × V",
            dims={"rows": rows, "seq": seq, "vocab": vocab},
            dtype=compute,
            note="autocast lm_head 출력 (accelerate가 fp32로 변환하기 전 순간).",
        ),
        spec(
            "logits.policy.fp32",
            LOSS,
            4 * nv,
            live_at=tps,
            formula="dpo-logits",
            shape="2B × T × V",
            dims={"rows": rows, "seq": seq, "vocab": vocab},
            dtype="float32",
            saved=True,
            note="accelerate native AMP의 fp32 출력, fused log-prob kernel이 backward용으로 저장.",
        ),
        spec(
            "loss.dpo.metric_masked_logits",
            LOSS,
            4 * (rows // 2) * max(seq - 1, 1) * vocab,
            live_at=[loss],
            formula="dpo-logits",
            shape="B × (T−1) × V",
            dtype="float32",
            note="logits/chosen·rejected metric의 boolean-index 복사 "
            "(branch별 순차, 상한: 전 위치).",
        ),
        spec(
            "loss.dpo.kernel_row_outputs",
            LOSS,
            4 * 4 * shifted,
            live_at=[loss, loss_bwd],
            formula="dpo-logits",
            shape="4 × 2B × (T−1)",
            dtype="float32",
        ),
        spec(
            "loss.dpo.grad_logits",
            LOSS,
            4 * shifted * vocab,
            live_at=[loss_bwd],
            formula="dpo-logits",
            shape="2B × (T−1) × V",
            dtype="float32",
            note="fused kernel backward의 empty_like(logits); "
            "slice·cast backward는 그 뒤에 순차 실행됩니다.",
        ),
    )
    if ref_fwd is not None:
        b.add(
            spec(
                "logits.reference.lm_head_output",
                LOSS,
                tensor_bytes(nv, compute),
                live_at=[ref_fwd],
                formula="dpo-logits",
                shape="2B × T × V",
                dtype=compute,
            ),
            spec(
                "logits.reference.fp32",
                LOSS,
                4 * nv,
                live_at=[ref_fwd, loss],
                formula="dpo-logits",
                shape="2B × T × V",
                dtype="float32",
                note="ref_outputs는 _compute_loss가 끝날 때까지 정책 logits와 함께 생존.",
            ),
        )
    lm_head_input_saved(b, cfg, positions, hidden, tps)


class DpoTrainer:
    trainer_id = "trl-1.14.1-dpo"
    objective = Objective.DPO

    def build_schedule(
        self,
        inventory: ModelInventory,
        cfg: ResolvedConfig,
        arch: ArchitectureAdapter,
        shape: BatchShape,
        plan: BatchPlan,
        scope: ScopeConfig,
    ) -> TrainingSchedule:
        if cfg.dpo is None:
            raise ValueError("DPO schedule needs cfg.dpo")
        strategy = cfg.dpo.reference_strategy
        precompute = strategy is ReferenceStrategy.PRECOMPUTED_LOG_PROBS
        standalone = strategy is ReferenceStrategy.STANDALONE_MODEL
        b = ScheduleBuilder()
        pfb = Phase.POLICY_FORWARD_BACKWARD
        load = plan_model_load(b, "policy", "정책 모델")
        ref_load = plan_model_load(b, "reference", "reference 모델") if standalone else None
        pre = (
            b.tp(
                Phase.REFERENCE_PRECOMPUTE,
                "forward",
                "Trainer __init__의 reference log-prob 사전 계산 (no-grad, 정책 wrapper 이전)",
            )
            if precompute
            else None
        )
        fwd = b.tp(pfb, "forward", "정책 forward 끝 (chosen B행 + rejected B행)")
        ref_fwd = (
            None
            if precompute
            else b.tp(
                pfb, "reference_forward", "reference no-grad forward (정책 graph·logits 생존)"
            )
        )
        loss = b.tp(pfb, "loss", "loss·metric 계산 (정책·reference fp32 logits 공존)")
        loss_bwd = b.tp(pfb, "loss_backward", "fused log-prob kernel·LM head backward")
        bwd = b.tp(pfb, "backward", "첫 decoder layer 재계산 backward (최대 작업 집합)")
        step = b.tp(Phase.OPTIMIZER_STEP, "step", "gradient clipping과 optimizer update")
        eval_tp, save_tp = add_scope_phases(b, scope)

        add_model_weights(b, arch, inventory, cfg, load)
        if ref_load is not None:
            self._reference_weights(b, arch, inventory, cfg, ref_load)
        rows, seq = shape.sequences_per_forward, shape.padded_length
        hidden, vocab = arch.lm_head_dims(inventory)
        if pre is not None:
            pairs = cfg.dpo.precompute_batch_size or cfg.microbatch
            pre_rows = 2 * pairs
            b.add(
                arch.no_grad_forward_ledger(
                    inventory, cfg, SequenceShape(batch=pre_rows, seq_len=seq), [pre], "precompute"
                ),
                spec(
                    "logits.precompute",
                    LOSS,
                    tensor_bytes(pre_rows * seq * vocab, cfg.load_dtype),
                    live_at=[pre],
                    formula="dpo-reference",
                    shape="2B_pre × T × V",
                    dims={"rows": pre_rows, "seq": seq, "vocab": vocab},
                    dtype=cfg.load_dtype,
                    note="사전 계산은 accelerate wrapper 이전이라 모델 dtype logits (no-grad).",
                ),
            )
        acts = arch.train_step_ledger(
            inventory,
            cfg,
            SequenceShape(batch=rows, seq_len=seq),
            StepTimepoints(forward=fwd, loss=loss, backward=bwd),
            POLICY_PREFIX,
        )
        b.add(extend_live_at(acts, loss, [t for t in (ref_fwd, loss_bwd) if t]))
        if ref_fwd is not None:
            b.add(
                arch.no_grad_forward_ledger(
                    inventory,
                    cfg,
                    SequenceShape(batch=rows, seq_len=seq),
                    [ref_fwd],
                    "reference_forward",
                )
            )
        policy_tps = [t for t in (fwd, ref_fwd, loss, loss_bwd) if t]
        add_dpo_logits(b, cfg, hidden, vocab, shape, policy_tps, ref_fwd, loss, loss_bwd)
        add_trainable_state(
            b,
            arch,
            inventory,
            cfg,
            weights_live=b.after(load.peak),
            grads_live=grads_live(cfg, [*policy_tps, bwd], bwd, step),
            states_live=b.step_ids(),
            step_tp=step,
        )
        add_scope_unknowns(b, eval_tp, save_tp)
        common_assumptions(b, cfg)
        b.assume(
            "dpo_reference",
            f"reference 전략: {strategy.value}.",
            "docs/research/trl-sft-dpo.md implications E",
        )
        add_workspace(b, cfg)
        return b.build()

    @staticmethod
    def _reference_weights(
        b: ScheduleBuilder,
        arch: ArchitectureAdapter,
        inventory: ModelInventory,
        cfg: ResolvedConfig,
        ref_load,
    ) -> None:
        assert cfg.dpo is not None
        if cfg.dpo.reference_model:  # a different checkpoint, not inspected here
            note = "별도 reference checkpoint의 inventory를 분석하지 않아 크기를 알 수 없습니다."
            b.add(
                unknown(
                    "reference.weights",
                    AllocationCategory.WEIGHTS_OTHER_MODELS,
                    live_at=b.from_(ref_load.peak),
                    formula="dpo-reference",
                    note=note,
                ),
                unknown(
                    "reference.device_map_budget",
                    AllocationCategory.LOAD_TRANSIENT,
                    live_at=[ref_load.check],
                    formula="load-phase",
                    note=note,
                ),
            )
            return
        # Same checkpoint, same model_init_kwargs and quantization_config as the policy.
        add_model_weights(
            b,
            arch,
            inventory,
            cfg,
            ref_load,
            prefix="reference",
            category=AllocationCategory.WEIGHTS_OTHER_MODELS,
        )
