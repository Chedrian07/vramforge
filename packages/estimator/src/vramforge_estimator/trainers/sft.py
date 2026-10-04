"""TRL 1.14.1 SFTTrainer schedule (docs/methodology.md#sft).

One step: model load, then per micro-step forward -> LM head/loss -> loss backward -> layer
backward, then the optimizer step. The default loss is TRL's `chunked_nll`: the LM head runs on
256 valid positions at a time inside a checkpoint, so no (B, T, V) logits exist and the loss peak
does not grow with the sequence length (docs/research/trl-sft-dpo.md §5.1). `hf_ce` (loss_type
"nll") materializes full fp32 logits (docs/research/architecture-memory.md §6.2).
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
    ResolvedConfig,
    ScopeConfig,
    SequenceShape,
)
from vramforge_estimator.units import dtype_bytes, tensor_bytes

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
    grads_live,
    lm_head_input_saved,
    lm_head_param_dtype,
    lm_head_trainable,
    plan_model_load,
    residual_dtype,
    scaled,
    spec,
    unknown,
)

CHUNK = 256  # trl/trainer/sft_trainer.py _CHUNKED_LM_HEAD_CHUNK_SIZE
LOSS = AllocationCategory.LOGITS_AND_LOSS


def add_sft_loss(
    b: ScheduleBuilder,
    cfg: ResolvedConfig,
    hidden: int,
    vocab: int,
    shape: BatchShape,
    fwd: str,
    loss: str,
    loss_bwd: str,
) -> None:
    rows = shape.sequences_per_forward
    seq = shape.padded_length
    shifted = rows * max(seq - 1, 1)
    res = residual_dtype(cfg)
    if cfg.loss_path == "trl_chunked_nll":
        cv = CHUNK * vocab
        gather = tensor_bytes(shifted * hidden, res)
        b.add(
            spec(
                "loss.chunked_nll.hidden_gather",
                LOSS,
                gather,
                live_at=[loss, loss_bwd],
                formula="sft-chunked-nll",
                shape="B·(T−1) × H",
                dims={"rows": shifted, "hidden": hidden},
                dtype=res,
                saved=True,
                note="valid 위치를 앞으로 모은 hidden 복사본 "
                "(chunk checkpoint 입력, backward까지 생존).",
            ),
            spec(
                "loss.chunked_nll.index_labels",
                LOSS,
                16 * shifted,
                live_at=[loss, loss_bwd],
                formula="sft-chunked-nll",
                shape="B·(T−1) × 2 int64",
                dtype="int64",
            ),
            spec(
                "loss.chunked_nll.chunk_forward",
                LOSS,
                scaled("15.0", cv),
                scaled("16.7", cv),
                live_at=[loss],
                formula="sft-chunked-nll",
                shape="256 × V",
                dims={"chunk": CHUNK, "vocab": vocab},
                note="chunk 하나의 logits·log-softmax·entropy 임시값 (CPU 측정 15.0–16.7 B/원소, "
                "chunk 수와 무관).",
            ),
        )
        if rows > 1:
            b.add(
                spec(
                    "loss.chunked_nll.reshape_copy",
                    LOSS,
                    gather,
                    live_at=[loss],
                    formula="sft-chunked-nll",
                    shape="B·(T−1) × H",
                    dtype=res,
                    note="B>1이면 hidden[..., :-1, :].reshape가 연속 복사본을 만든 뒤 정렬합니다.",
                )
            )
        if lm_head_trainable(cfg):
            vh = vocab * hidden
            grad = tensor_bytes(vh, lm_head_param_dtype(cfg))
            b.add(
                spec(
                    "loss.chunked_nll.lm_head_grad_accumulation",
                    LOSS,
                    3 * grad,
                    3 * grad + scaled("18", cv),
                    live_at=[loss_bwd],
                    formula="sft-chunked-nll",
                    shape="3 × V × H",
                    dims={"vocab": vocab, "hidden": hidden},
                    note="학습되는 lm_head: chunk별 [V,H] weight grad가 out-of-place로 누적되어 "
                    "3벌이 공존합니다 (CPU 검증).",
                )
            )
        else:
            b.add(
                spec(
                    "loss.chunked_nll.chunk_backward",
                    LOSS,
                    scaled("16.0", cv),
                    scaled("18.0", cv),
                    live_at=[loss_bwd],
                    formula="sft-chunked-nll",
                    shape="256 × V",
                    dims={"chunk": CHUNK, "vocab": vocab},
                    note="chunk 재계산 backward 임시값 (CPU 측정 16.0–17.7 B/원소, 보수 상한 18).",
                )
            )
        if dtype_bytes(res) > dtype_bytes(cfg.effective_dtypes.compute):
            b.add(
                spec(
                    "loss.chunked_nll.lm_head_weight_cast",
                    LOSS,
                    tensor_bytes(vocab * hidden, cfg.effective_dtypes.compute),
                    live_at=[loss, loss_bwd],
                    formula="sft-chunked-nll",
                    shape="V × H",
                    dtype=cfg.effective_dtypes.compute,
                    note="fp32 로드 + bf16 autocast: chunk matmul마다 lm_head weight의 bf16 사본 "
                    "(소스 기반 추론).",
                )
            )
    elif cfg.loss_path == "hf_ce":
        nv = rows * seq * vocab
        b.add(
            spec(
                "loss.hf_ce.forward",
                LOSS,
                10 * nv,
                12 * nv,
                live_at=[loss],
                formula="sft-nll",
                shape="B·T × V",
                dims={"positions": rows * seq, "vocab": vocab},
                note="bf16 logits 2 + fp32 복사 4 + log_softmax 4 "
                "(+ accelerate fp32 출력·metric 복사).",
            ),
            spec(
                "loss.hf_ce.backward",
                LOSS,
                12 * nv,
                14 * nv,
                live_at=[loss_bwd],
                formula="sft-nll",
                shape="B·T × V",
                dims={"positions": rows * seq, "vocab": vocab},
                note="저장된 log_softmax 4 + grad 8 (logits 참조가 남으면 14).",
            ),
        )
        lm_head_input_saved(b, cfg, rows * seq, hidden, [fwd, loss, loss_bwd])
    else:
        b.add(
            unknown(
                "loss.unsupported_path",
                LOSS,
                live_at=[loss, loss_bwd],
                formula="sft",
                note=f"loss 경로 {cfg.loss_path!r}의 메모리 모델이 없습니다.",
            )
        )


class SftTrainer:
    trainer_id = "trl-1.14.1-sft"
    objective = Objective.SFT

    def build_schedule(
        self,
        inventory: ModelInventory,
        cfg: ResolvedConfig,
        arch: ArchitectureAdapter,
        shape: BatchShape,
        plan: BatchPlan,
        scope: ScopeConfig,
    ) -> TrainingSchedule:
        b = ScheduleBuilder()
        pfb = Phase.POLICY_FORWARD_BACKWARD
        load = plan_model_load(b, "policy", "정책 모델")
        fwd = b.tp(pfb, "forward", "정책 forward 끝: 저장된 activation이 모두 생존")
        loss = b.tp(pfb, "loss", "LM head·loss forward")
        loss_bwd = b.tp(pfb, "loss_backward", "loss·LM head backward (저장된 activation 생존)")
        bwd = b.tp(pfb, "backward", "첫 decoder layer 재계산 backward (최대 작업 집합)")
        step = b.tp(Phase.OPTIMIZER_STEP, "step", "gradient clipping과 optimizer update")
        eval_tp, save_tp = add_scope_phases(b, scope)

        add_model_weights(b, arch, inventory, cfg, load)
        seq = SequenceShape(batch=shape.sequences_per_forward, seq_len=shape.padded_length)
        acts = arch.train_step_ledger(
            inventory, cfg, seq, StepTimepoints(forward=fwd, loss=loss, backward=bwd), "policy"
        )
        b.add(extend_live_at(acts, loss, [loss_bwd]))
        hidden, vocab = arch.lm_head_dims(inventory)
        add_sft_loss(b, cfg, hidden, vocab, shape, fwd, loss, loss_bwd)
        fb = [fwd, loss, loss_bwd, bwd]
        add_trainable_state(
            b,
            arch,
            inventory,
            cfg,
            weights_live=b.from_(fwd),
            grads_live=grads_live(cfg, fb, bwd, step),
            states_live=b.step_ids(),
            step_tp=step,
        )
        add_scope_unknowns(b, eval_tp, save_tp)
        common_assumptions(b, cfg)
        add_workspace(b, cfg)
        return b.build()
