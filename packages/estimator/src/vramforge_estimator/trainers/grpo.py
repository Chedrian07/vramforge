"""TRL 1.14.1 GRPOTrainer schedule (docs/methodology.md#grpo).

Per generation: one `generate` over the C = B_update x spg live sequences of the device
(no prefix sharing), optional old/reference log-prob passes, the reward, then spg x num_iterations
training micro-steps over the buffered rollout. Order and conditions follow
docs/research/trl-grpo.md §10 and implications R1-R8:

- old log-probs: K % (spg x num_iterations) != 0 (vLLM importance sampling is unsupported here);
- reference log-probs: beta != 0 (PEFT: adapter disabled; full fine-tuning: a second model);
- gradients are alive during a rollout iff (spg x num_iterations) % K != 0;
- the reward has no GPU footprint unless a local reward model sits on the training GPU, in which
  case its size is unknown until that model is inspected.
"""

from __future__ import annotations

from vramforge_estimator.architectures import (
    ArchitectureAdapter,
    GenerationTimepoints,
    StepTimepoints,
)
from vramforge_estimator.schemas import (
    AllocationCategory,
    BatchPlan,
    BatchShape,
    ErrorCode,
    ExcludedComponent,
    GrpoResolved,
    ModelInventory,
    Objective,
    Phase,
    ResolvedConfig,
    RewardKind,
    ScopeConfig,
    SequenceShape,
    Strategy,
)
from vramforge_estimator.units import tensor_bytes

from .base import TrainingSchedule
from .common import (
    ScheduleBuilder,
    add_model_weights,
    add_scope_phases,
    add_scope_unknowns,
    add_trainable_state,
    add_workspace,
    common_assumptions,
    extend_live_at,
    lm_head_input_saved,
    plan_model_load,
    resolution_value,
    spec,
    unknown,
)

LOSS = AllocationCategory.LOGITS_AND_LOSS
REWARD_EXCLUSION = {
    RewardKind.UNSPECIFIED: (
        "reward가 지정되지 않아 reward 메모리를 계산에서 제외했습니다. "
        "실행에는 reward가 필요합니다 (조건부 결과).",
        ErrorCode.GRPO_REWARD_UNSPECIFIED,
    ),
    RewardKind.CPU_RULE: (
        "CPU reward 함수는 학습 GPU를 쓰지 않는 것으로 계산했습니다. "
        "CPU·RAM 사용량은 포함하지 않습니다.",
        None,
    ),
    RewardKind.REMOTE: (
        "원격 reward 서버의 자원은 이 GPU 계산에 포함하지 않습니다 "
        "(원격 자원이 0이라는 뜻이 아닙니다).",
        None,
    ),
    RewardKind.LOCAL_MODEL: (
        "reward 모델이 다른 장치에 있다고 지정되어 이 GPU 계산에서 제외했습니다.",
        None,
    ),
}


def grpo_flags(g: GrpoResolved) -> tuple[bool, bool, bool]:
    """(old log-prob pass, reference log-prob pass, gradients alive during rollout)."""
    generate_every = g.steps_per_generation * g.num_iterations
    old = g.accumulation % generate_every != 0
    return old, g.reference_needed, generate_every % g.accumulation != 0


def rollout_buffer_bytes(live: int, prompt: int, completion: int, logp_tensors: int) -> int:
    """ids + masks (int64) per prompt/completion position, fp32 log-prob rows, advantages
    (docs/research/trl-grpo.md §5.4)."""
    return live * (prompt * 16 + completion * (16 + 4 * logp_tensors)) + live * 4


class GrpoTrainer:
    trainer_id = "trl-1.14.1-grpo"
    objective = Objective.GRPO

    def build_schedule(
        self,
        inventory: ModelInventory,
        cfg: ResolvedConfig,
        arch: ArchitectureAdapter,
        shape: BatchShape,
        plan: BatchPlan,
        scope: ScopeConfig,
    ) -> TrainingSchedule:
        g = cfg.grpo
        if g is None:
            raise ValueError("GRPO schedule needs cfg.grpo")
        old, ref, grads_at_rollout = grpo_flags(g)
        peft = cfg.strategy is not Strategy.FULL
        ref_model = ref and not peft
        reward_on_gpu = g.reward_kind is RewardKind.LOCAL_MODEL and bool(
            resolution_value(cfg, "grpo.reward.on_training_gpu", True)
        )
        b = ScheduleBuilder()
        pfb = Phase.POLICY_FORWARD_BACKWARD
        rollout = Phase.ROLLOUT_PREFILL_AND_DECODE
        load = plan_model_load(b, "policy", "정책 모델")
        ref_load = plan_model_load(b, "reference", "reference 모델") if ref_model else None
        prefill = b.tp(rollout, "prefill", "C개 sequence의 prompt prefill (no-grad, GC 꺼짐)")
        decode = b.tp(rollout, "decode", "마지막 decode step: cache가 P + L − 1 위치까지 성장")
        old_tp = (
            b.tp(pfb, "old_logprob", "old policy log-prob no-grad pass (B_update행씩)")
            if old
            else None
        )
        ref_tp = (
            b.tp(pfb, "reference_logprob", "reference log-prob no-grad pass (B_update행씩)")
            if ref
            else None
        )
        reward_tp = (
            b.tp(Phase.REWARD, "score", "local reward 모델 forward") if reward_on_gpu else None
        )
        fwd = b.tp(pfb, "forward", "학습 micro-step forward 끝")
        loss = b.tp(pfb, "loss", "fused log-prob/entropy kernel forward")
        loss_bwd = b.tp(pfb, "loss_backward", "fused kernel·LM head backward")
        bwd = b.tp(pfb, "backward", "첫 decoder layer 재계산 backward (최대 작업 집합)")
        step = b.tp(Phase.OPTIMIZER_STEP, "step", "gradient clipping과 optimizer update")
        eval_tp, save_tp = add_scope_phases(b, scope)
        if not reward_on_gpu:
            reason, code = REWARD_EXCLUSION[g.reward_kind]
            b.excluded.append(ExcludedComponent(name=Phase.REWARD.value, reason=reason, code=code))

        add_model_weights(b, arch, inventory, cfg, load)
        if ref_load is not None:
            # beta != 0 without PEFT: create_model_from_path with the policy's kwargs and
            # quantization_config (docs/research/trl-grpo.md §8).
            add_model_weights(
                b,
                arch,
                inventory,
                cfg,
                ref_load,
                prefix="reference",
                category=AllocationCategory.WEIGHTS_OTHER_MODELS,
            )
        if reward_tp is not None:
            b.add(
                unknown(
                    "reward_model.weights",
                    AllocationCategory.WEIGHTS_OTHER_MODELS,
                    live_at=b.after(load.peak),
                    formula="grpo-reward",
                    note="local reward 모델의 inventory를 분석하지 않아 크기를 알 수 없습니다.",
                ),
                unknown(
                    "reward_model.forward",
                    AllocationCategory.RECOMPUTE_WORKING_SET,
                    live_at=[reward_tp],
                    formula="grpo-reward",
                    note="reward 모델의 C × T_reward no-grad forward는 "
                    "그 모델 구조가 있어야 계산됩니다.",
                ),
            )

        prompt, completion = self._lengths(shape)
        if prompt is None or completion is None:
            b.add(
                unknown(
                    "grpo.shape",
                    LOSS,
                    live_at=b.after(load.peak),
                    formula="grpo",
                    note="batch 계획에 prompt/completion 길이가 없어 "
                    "rollout·update를 계산할 수 없습니다.",
                )
            )
        else:
            self._rollout(b, arch, inventory, cfg, g, prompt, completion, prefill, decode)
            self._logprob_passes(b, arch, inventory, cfg, g, prompt, completion, old_tp, ref_tp)
            self._update(b, arch, inventory, cfg, g, prompt, completion, fwd, loss, loss_bwd, bwd)
            logps = int(old) + int(ref)
            buffers = rollout_buffer_bytes(g.live_sequences, prompt, completion, logps)
            generation = [t for t in (old_tp, ref_tp, reward_tp) if t]
            b.add(
                spec(
                    "rollout_buffers.previous",
                    AllocationCategory.ROLLOUT_BUFFERS,
                    buffers,
                    live_at=[prefill, decode, *generation],
                    formula="grpo-rollout-buffers",
                    note="이전 generation의 _buffered_inputs는 새 rollout이 끝날 때까지 남습니다.",
                ),
                spec(
                    "rollout_buffers.current",
                    AllocationCategory.ROLLOUT_BUFFERS,
                    buffers,
                    live_at=[*generation, fwd, loss, loss_bwd, bwd, step],
                    formula="grpo-rollout-buffers",
                    note="prompt/completion id·mask, log-prob, advantage (spg × num_iterations "
                    "micro-step 동안 생존).",
                ),
            )
        fb = [fwd, loss, loss_bwd, bwd]
        window = fb if g.accumulation > 1 else [bwd]
        rollout_tps = [prefill, decode, *[t for t in (old_tp, ref_tp, reward_tp) if t]]
        grads = [*(rollout_tps if grads_at_rollout else []), *window, step]
        add_trainable_state(
            b,
            arch,
            inventory,
            cfg,
            weights_live=b.after(load.peak),
            grads_live=grads,
            states_live=b.step_ids(),
            step_tp=step,
        )
        add_scope_unknowns(b, eval_tp, save_tp)
        common_assumptions(b, cfg)
        b.assume(
            "grpo_flags",
            f"C={g.live_sequences}, B_update={g.update_microbatch}, K={g.accumulation}, "
            f"spg={g.steps_per_generation}, num_iterations={g.num_iterations}: "
            f"old log-prob {'있음' if old else '없음'}, "
            f"reference log-prob {'있음' if ref else '없음'}, "
            f"rollout 중 gradient {'있음' if grads_at_rollout else '없음'}.",
            "docs/research/trl-grpo.md implications R1",
        )
        b.assume(
            "grpo_fused_kernel",
            "Linux CUDA의 TRL Triton fused log-prob kernel 경로(logits 6E/4E/8E, no-grad 10E)를 "
            "가정합니다.",
            "docs/research/trl-grpo.md implications R4",
        )
        add_workspace(b, cfg)
        return b.build()

    @staticmethod
    def _lengths(shape: BatchShape) -> tuple[int | None, int | None]:
        completion = shape.completion_length
        prompt = shape.prompt_length
        if prompt is None and completion is not None:
            prompt = shape.padded_length - completion
        return prompt, completion

    @staticmethod
    def _logits_bytes_per_step(cfg: ResolvedConfig, live: int, vocab: int) -> int:
        # PEFT generate bypasses the accelerate wrapper (model dtype logits); a full model keeps
        # the fp32 output conversion (docs/research/trl-grpo.md §4.4, table A/F/C/G/Q/R).
        dtype = "float32" if cfg.strategy is Strategy.FULL else cfg.load_dtype
        return tensor_bytes(live * vocab, dtype)

    def _rollout(self, b, arch, inventory, cfg, g, prompt, completion, prefill, decode) -> None:
        _, vocab = arch.lm_head_dims(inventory)
        b.add(
            arch.generation_ledger(
                inventory,
                cfg,
                g.live_sequences,
                prompt,
                completion,
                GenerationTimepoints(prefill=prefill, decode=decode),
                "rollout",
            ),
            spec(
                "logits.rollout.step_output",
                LOSS,
                self._logits_bytes_per_step(cfg, g.live_sequences, vocab),
                live_at=[prefill, decode],
                formula="grpo-rollout",
                shape="C × 1 × V",
                dims={"sequences": g.live_sequences, "vocab": vocab},
                note="prefill은 logits_to_keep=1, decode는 step마다 (C,1,V).",
            ),
            spec(
                "logits.rollout.fp32_copy",
                LOSS,
                4 * g.live_sequences * vocab,
                live_at=[prefill, decode],
                formula="grpo-rollout",
                shape="C × V",
                dtype="float32",
                note="generate가 마지막 위치 logits를 fp32로 복사합니다.",
            ),
        )

    def _logprob_passes(self, b, arch, inventory, cfg, g, prompt, completion, old_tp, ref_tp):
        _, vocab = arch.lm_head_dims(inventory)
        rows = g.update_microbatch
        elements = rows * (completion + 1) * vocab
        chunks = g.live_sequences // rows
        factor = 10 if chunks >= 2 else 6
        seq = SequenceShape(
            batch=rows, seq_len=prompt + completion, logits_positions_per_sequence=completion + 1
        )
        for tp, label in ((old_tp, "old_logprob"), (ref_tp, "reference_logprob")):
            if tp is None:
                continue
            b.add(
                arch.no_grad_forward_ledger(inventory, cfg, seq, [tp], label),
                spec(
                    f"logits.{label}",
                    LOSS,
                    factor * elements,
                    live_at=[tp],
                    formula="grpo-logprob-passes",
                    shape="B_update × (L+1) × V",
                    dims={"rows": rows, "positions": completion + 1, "vocab": vocab},
                    note=f"no-grad pass {chunks}개 chunk: {factor} B/원소 (CPU 에뮬레이션 측정).",
                ),
            )

    def _update(self, b, arch, inventory, cfg, g, prompt, completion, fwd, loss, loss_bwd, bwd):
        hidden, vocab = arch.lm_head_dims(inventory)
        rows = g.update_microbatch
        seq = SequenceShape(
            batch=rows, seq_len=prompt + completion, logits_positions_per_sequence=completion + 1
        )
        acts = arch.train_step_ledger(
            inventory, cfg, seq, StepTimepoints(forward=fwd, loss=loss, backward=bwd), "policy"
        )
        b.add(extend_live_at(acts, loss, [loss_bwd]))
        if cfg.loss_path != "trl_fused_logprob":
            b.add(
                unknown(
                    "loss.unsupported_path",
                    LOSS,
                    live_at=[loss, loss_bwd],
                    formula="grpo-policy-logits",
                    note=f"loss 경로 {cfg.loss_path!r}의 메모리 모델이 없습니다.",
                )
            )
            return
        elements = rows * (completion + 1) * vocab
        compute = cfg.effective_dtypes.compute
        b.add(
            spec(
                "logits.policy.lm_head_output",
                LOSS,
                tensor_bytes(elements, compute),
                live_at=[fwd],
                formula="grpo-policy-logits",
                shape="B_update × (L+1) × V",
                dtype=compute,
                note="autocast lm_head 출력 (fp32 변환 전 순간).",
            ),
            spec(
                "logits.policy.fp32",
                LOSS,
                4 * elements,
                live_at=[fwd, loss, loss_bwd],
                formula="grpo-policy-logits",
                shape="B_update × (L+1) × V",
                dtype="float32",
                saved=True,
                note="accelerate fp32 출력, fused kernel이 backward용으로 저장.",
            ),
            spec(
                "loss.grpo.grad_logits",
                LOSS,
                4 * rows * completion * vocab,
                live_at=[loss_bwd],
                formula="grpo-policy-logits",
                shape="B_update × L × V",
                dtype="float32",
                note="fused kernel backward의 empty_like(logits).",
            ),
        )
        # logits_to_keep slices a view of the final hidden states, so a trainable lm_head keeps
        # the whole (B, P+L, H) storage alive.
        lm_head_input_saved(b, cfg, rows * (prompt + completion), hidden, [fwd, loss, loss_bwd])
