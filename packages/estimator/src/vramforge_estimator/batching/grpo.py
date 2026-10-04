"""GRPO batch arithmetic of TRL 1.14.1 (docs/research/trl-grpo.md §2.1, §3, R1).

Single process (W = 1): ``generation_batch_size = B_update x steps_per_generation``,
``U = generation_batch_size / num_generations`` unique prompts per generation,
``C = B_update x steps_per_generation`` sequences generated together. ``RepeatSampler`` drops the
last incomplete chunk of U prompts every epoch, so ``N mod U`` prompts are skipped per epoch
(trl trainer/utils.py:890-912).
"""

from __future__ import annotations

from dataclasses import dataclass

from vramforge_estimator.errors import EstimatorError
from vramforge_estimator.schemas import ErrorCode, GrpoResolved, Issue, Severity, Stage


@dataclass(frozen=True)
class GrpoLayout:
    num_generations: int  # G
    generation_batch_size: int  # global, = U x G
    steps_per_generation: int
    update_microbatch: int  # B_update (per_device_train_batch_size)
    accumulation: int  # K
    num_iterations: int
    unique_prompts: int  # U
    live_sequences: int  # C = B_update x steps_per_generation

    def dropped_rows(self, rows: int) -> int:
        """Prompts RepeatSampler skips per epoch for a dataset of `rows` prompts."""
        return rows % self.unique_prompts


def _constraint(message: str, **details: object) -> EstimatorError:
    return EstimatorError(
        Issue(
            code=ErrorCode.TRAINER_BATCH_CONSTRAINT,
            severity=Severity.ERROR,
            stage=Stage.PLANNING_BATCHES,
            user_message=message,
            details=dict(details),
        )
    )


def resolve_layout(grpo: GrpoResolved) -> GrpoLayout:
    """Validate the TRL 1.14.1 relations GRPOTrainer enforces and derive U and C. Configurations
    TRL would reject raise TRAINER_BATCH_CONSTRAINT (grpo_config.py:1083-1128)."""
    g, gbs = grpo.num_generations, grpo.generation_batch_size
    b, spg = grpo.update_microbatch, grpo.steps_per_generation
    if min(g, gbs, b, spg, grpo.accumulation, grpo.num_iterations) < 1:
        raise _constraint("GRPO batch 설정 값은 모두 1 이상이어야 합니다.")
    if g < 2:
        raise _constraint(
            "TRL GRPO는 prompt당 생성 수(num_generations)가 2 이상이어야 합니다.", num_generations=g
        )
    if gbs % g:
        raise _constraint(
            f"generation_batch_size({gbs})가 num_generations({g})로 나누어떨어지지 않아 TRL이 "
            "실행을 거부합니다.",
            generation_batch_size=gbs,
            num_generations=g,
        )
    if gbs != b * spg:
        raise _constraint(
            f"generation_batch_size({gbs})는 per_device_train_batch_size({b}) × "
            f"steps_per_generation({spg})와 같아야 합니다(단일 GPU).",
            generation_batch_size=gbs,
            update_microbatch=b,
            steps_per_generation=spg,
        )
    return GrpoLayout(
        num_generations=g,
        generation_batch_size=gbs,
        steps_per_generation=spg,
        update_microbatch=b,
        accumulation=grpo.accumulation,
        num_iterations=grpo.num_iterations,
        unique_prompts=gbs // g,
        live_sequences=b * spg,
    )


__all__ = ["GrpoLayout", "resolve_layout"]
