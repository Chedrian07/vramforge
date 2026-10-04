"""GRPO batch resolution with the exact TRL 1.14.1 rules (docs/research/trl-grpo.md §2.1, R1).

spg = K                       (generation_batch_size and steps_per_generation both unset)
spg = gbs / (B_update x W)    (gbs set; must divide exactly)
gbs = B_update x W x spg      (spg set, or both unset)
gbs % G == 0, G >= 2, gbs and spg are mutually exclusive
U = gbs / G,  C = B_update x spg = gbs / W
old log-prob pass  <=> K % (spg x num_iterations) != 0
gradients at rollout <=> (spg x num_iterations) % K != 0
"""

from __future__ import annotations

from dataclasses import dataclass

from vramforge_estimator.errors import make_issue
from vramforge_estimator.schemas import ErrorCode, Issue, Stage


@dataclass(frozen=True)
class GrpoBatch:
    generation_batch_size: int  # global (all processes)
    steps_per_generation: int
    unique_prompts: int  # U
    live_sequences: int  # C (per device)
    old_logprob_pass: bool
    grads_alive_during_rollout: bool


def _constraint(message: str, trl_message: str, component: str, **details: object) -> Issue:
    issue = make_issue(
        ErrorCode.TRAINER_BATCH_CONSTRAINT, message, stage=Stage.REQUEST, component=component
    )
    issue.details.update(trl_message=trl_message, **details)
    return issue


def resolve_grpo_batch(
    *,
    num_generations: int,
    microbatch: int,
    accumulation: int,
    generation_batch_size: int | None,
    steps_per_generation: int | None,
    num_iterations: int = 1,
    num_processes: int = 1,
) -> tuple[GrpoBatch | None, list[Issue]]:
    g, b, k, w = num_generations, microbatch, accumulation, num_processes
    gbs, spg = generation_batch_size, steps_per_generation
    if gbs is not None and spg is not None:
        return None, [
            _constraint(
                "generation_batch_size와 steps_per_generation은 동시에 지정할 수 없습니다.",
                "'generation_batch_size' and 'steps_per_generation' can not be both configured "
                "at the same time",
                "grpo.steps_per_generation",
            )
        ]
    if gbs is None and spg is None:
        spg = k
        gbs = b * w * spg
    elif gbs is not None:
        if gbs % (b * w) != 0:
            return None, [
                _constraint(
                    f"generation_batch_size({gbs})는 전체 update batch({b * w})로 나누어떨어져야 "
                    "합니다.",
                    f"generation_batch_size ({gbs}) must be divisible by the global batch size "
                    f"({b * w}).",
                    "grpo.generation_batch_size",
                    generation_batch_size=gbs,
                    global_batch_size=b * w,
                )
            ]
        spg = gbs // (b * w)
    else:
        assert spg is not None
        gbs = b * w * spg
    issues: list[Issue] = []
    if gbs % g != 0:
        issues.append(
            _constraint(
                f"generation_batch_size({gbs})는 num_generations({g})로 나누어떨어져야 합니다.",
                f"generation_batch_size ({gbs}) must be divisible by num_generations ({g}).",
                "grpo.num_generations",
                generation_batch_size=gbs,
                num_generations=g,
            )
        )
    if g < 2:
        issues.append(
            _constraint(
                "GRPO는 prompt당 2개 이상의 generation이 필요합니다.",
                "GRPO requires at least 2 generations per prompt to calculate the advantages.",
                "grpo.num_generations",
                num_generations=g,
            )
        )
    if issues:
        return None, issues
    every = spg * num_iterations
    return (
        GrpoBatch(
            generation_batch_size=gbs,
            steps_per_generation=spg,
            unique_prompts=gbs // g,
            live_sequences=b * spg,
            old_logprob_pass=k % every != 0,
            grads_alive_during_rollout=every % k != 0,
        ),
        [],
    )
