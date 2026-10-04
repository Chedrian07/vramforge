"""Batch planning from the row-length table (plan.md §8).

Token-slot formulas follow the TRL 1.14.1 collators (docs/research/trl-sft-dpo.md §4, §8.1, C):

* SFT  ``DataCollatorForLanguageModeling``: B x round_up(max L_i, m), right padding.
* DPO  ``DataCollatorForPreference``: chosen block (B rows) then rejected block (B rows), all
  padded to the longest of both branches: 2B x round_up(max_i max(Lc_i, Lr_i), m).
* GRPO update microbatch: B_update sequences of width round_up(P, m) + round_up(L, m), where P
  and L are the generation-batch-wide maxima; the policy forward keeps L + 1 logits positions per
  sequence (docs/research/trl-grpo.md §6.4, R2, R4).

``worst_case`` is the structural conservative shape (the longest rows together). The seeded
sampler order needs torch's RNG, so ``sampler_max`` is only filled when the order cannot change
the shape (one row per batch, all rows in one batch, or one unique prompt per GRPO generation);
otherwise it stays None and an info issue says the structural worst case is used.
"""

from __future__ import annotations

import hashlib
from array import array
from collections.abc import Sequence
from heapq import nlargest

from vramforge_estimator import keys
from vramforge_estimator.errors import EstimatorError
from vramforge_estimator.scan import LengthTable
from vramforge_estimator.schemas import (
    BatchPlan,
    BatchShape,
    ErrorCode,
    GrpoBatchPlan,
    Issue,
    Objective,
    ResolvedConfig,
    SamplerPlan,
    Severity,
    Stage,
)
from vramforge_estimator.units import round_up

from .grpo import GrpoLayout, resolve_layout

PLANNER_VERSION = "1"
CHUNKED_LM_HEAD_CHUNK = 256  # trl sft_trainer.py:87 (_CHUNKED_LM_HEAD_CHUNK_SIZE)


def _issue(code: ErrorCode, severity: Severity, message: str, **details: object) -> Issue:
    return Issue(
        code=code,
        severity=severity,
        stage=Stage.PLANNING_BATCHES,
        user_message=message,
        details=dict(details),
    )


def _no_lengths() -> EstimatorError:
    return EstimatorError(
        _issue(
            ErrorCode.SCAN_FAILED_ROWS,
            Severity.ERROR,
            "성공적으로 토큰화한 row가 없어 batch를 계획할 수 없습니다.",
        )
    )


def _longest(values: Sequence[int | None], k: int) -> list[int]:
    """Indices of the k largest values (ties: earlier row first)."""
    present = [i for i, v in enumerate(values) if v is not None]
    return nlargest(k, present, key=lambda i: (values[i], -i))


def _lengths_digest(lengths: LengthTable) -> str:
    """Content key of the length table. Stands in for preprocess_key, which the planner's
    contract does not receive (reported as a change request)."""
    h = hashlib.sha256()
    for column in (
        lengths.prompt_tokens,
        lengths.completion_tokens,
        lengths.sequence_tokens,
        lengths.loss_token_count,
        lengths.chosen_total_tokens,
        lengths.rejected_total_tokens,
    ):
        for start in range(0, len(column), 65_536):
            chunk = column[start : start + 65_536]
            h.update(array("q", (-1 if v is None else v for v in chunk)).tobytes())
        h.update(b"|")
    for row_id in lengths.row_ids:
        h.update(row_id.encode("utf-8"))
        h.update(b"\n")
    return f"len_{h.hexdigest()}"


def _order_note(rows: int, batch: int, seed: int) -> str:
    tail = rows % batch
    if tail:
        return (
            f"RandomSampler(seed={seed}), dataloader_drop_last=False: 마지막 부분 batch({tail}개 "
            f"row)까지 {rows:,}개 row를 모두 사용합니다."
        )
    return (
        f"RandomSampler(seed={seed}), dataloader_drop_last=False: {rows:,}개 row가 batch "
        f"{batch}개 단위로 모두 사용됩니다."
    )


def _order_issue() -> Issue:
    return _issue(
        ErrorCode.TRAINER_BATCH_CONSTRAINT,
        Severity.INFO,
        "batch가 2개 이상의 row를 묶으면 실제 batch 구성이 seed에 따른 sampler 순서(torch RNG)에 "
        "달려 있어 재현하지 않았습니다. 가장 긴 row들을 한 batch로 묶은 구조적 최악 shape를 "
        "사용합니다.",
        reason="sampler_order_requires_torch_rng",
    )


def _as_sampler_max(shape: BatchShape, why: str) -> BatchShape:
    return shape.model_copy(
        update={"name": "sampler_max", "description": f"{shape.description} {why}"}
    )


def plan_batches(lengths: LengthTable, resolved: ResolvedConfig, *, seed: int) -> BatchPlan:
    """Build the worst-case shape, the seeded-sampler maximum and (GRPO) one shape per completion
    budget. Reports sampler coverage (rows dropped by batch/group divisibility) as an issue."""
    if resolved.microbatch < 1 or resolved.accumulation < 1:
        raise EstimatorError(
            _issue(
                ErrorCode.INVALID_REQUEST,
                Severity.ERROR,
                "microbatch와 gradient accumulation은 1 이상이어야 합니다.",
            )
        )
    if resolved.objective is Objective.GRPO:
        return _plan_grpo(lengths, resolved, seed)
    return _plan_pairs_or_samples(lengths, resolved, seed)


def _plan_pairs_or_samples(lengths: LengthTable, resolved: ResolvedConfig, seed: int) -> BatchPlan:
    objective = resolved.objective
    m = resolved.pad_to_multiple_of
    batch = resolved.microbatch
    if objective is Objective.SFT:
        per_row: list[int | None] = list(lengths.sequence_tokens)
    else:
        per_row = [
            max(v for v in (c, r) if v is not None) if c is not None or r is not None else None
            for c, r in zip(lengths.chosen_total_tokens, lengths.rejected_total_tokens, strict=True)
        ]
    rows = sum(1 for v in per_row if v is not None)
    if rows == 0:
        raise _no_lengths()
    b = min(batch, rows)
    top = _longest(per_row, b)
    longest = max(per_row[i] or 0 for i in top)
    width = round_up(longest, m)
    ids = [lengths.row_ids[i] for i in top]
    padding = f"pad_to_multiple_of={m}" if m else "pad_to_multiple_of 없음"
    if objective is Objective.SFT:
        sequences = b
        logits, logits_note = _sft_logits(lengths, resolved.loss_path, b, width)
        description = (
            f"SFT 구조적 최악: 가장 긴 {b}개 row를 한 batch로 묶어 "
            f"DataCollatorForLanguageModeling이 오른쪽 padding으로 {width} 토큰({padding})까지 "
            f"맞춘 {b}×{width}={b * width:,} token slot. loss가 없는 prompt 토큰과 padding도 "
            f"forward에 포함됩니다. {logits_note}"
        )
        unit = "samples"
        collator = "DataCollatorForLanguageModeling"
    else:
        sequences = 2 * b
        logits = sequences * width
        description = (
            f"DPO 구조적 최악: 가장 긴 {b}개 pair의 chosen {b}행 뒤에 rejected {b}행을 붙여"
            f"(DataCollatorForPreference) 두 branch 전체 최대 {width} 토큰({padding})까지 오른쪽 "
            f"padding한 {sequences}×{width}={sequences * width:,} token slot. prompt와 padding을 "
            "포함한 모든 위치의 logits를 계산합니다."
        )
        unit = "pairs"
        collator = "DataCollatorForPreference"
    worst = BatchShape(
        name="worst_case",
        objective=objective,
        rows_per_microbatch=b,
        sequences_per_forward=sequences,
        padded_length=width,
        token_slots=sequences * width,
        logits_positions=logits,
        source_row_ids=ids,
        description=description,
    )
    issues: list[Issue] = []
    if batch == 1 or rows <= batch:
        why = (
            "batch당 row가 1개라 seed 순서와 관계없이 이 shape가 그대로 나타납니다."
            if batch == 1
            else "전체 row가 한 batch에 들어가 순서와 관계없이 같은 shape입니다."
        )
        sampler_max: BatchShape | None = _as_sampler_max(worst, why)
    else:
        sampler_max = None
        issues.append(_order_issue())
    sampler = SamplerPlan(
        kind="random",
        seed=seed,
        drop_last=False,
        covers_all_rows=True,
        dropped_rows=0,
        note=_order_note(rows, batch, seed),
    )
    return BatchPlan(
        objective=objective,
        unit=unit,
        microbatch=batch,
        accumulation=resolved.accumulation,
        effective_batch=batch * resolved.accumulation,
        pad_to_multiple_of=m,
        padding_side="right",
        sampler=sampler,
        worst_case=worst,
        sampler_max=sampler_max,
        issues=issues,
        batch_key=_batch_key(lengths, resolved, seed, collator, sampler, batch),
    )


def _sft_logits(lengths: LengthTable, loss_path: str, b: int, width: int) -> tuple[int, str]:
    """LM-head positions per forward. chunked_nll projects only label positions, padded up to
    256-position chunks (trl sft_trainer.py:197-220); other paths project every position."""
    if "chunked" not in loss_path:
        return b * width, f"loss 경로 {loss_path}: 모든 위치({b}×{width})의 logits를 계산합니다."
    top_loss = [lengths.loss_token_count[i] for i in _longest(lengths.loss_token_count, b)]
    # unknown label counts: every non-first position of every row is the structural bound
    known = len(top_loss) == b
    valid = sum(v or 0 for v in top_loss) if known else b * max(0, width - 1)
    chunks = max(1, -(-valid // CHUNKED_LM_HEAD_CHUNK))
    positions = chunks * CHUNKED_LM_HEAD_CHUNK
    return positions, (
        f"chunked_nll: loss 위치 최대 {valid:,}개를 {CHUNKED_LM_HEAD_CHUNK}개 단위 chunk "
        f"{chunks}개로 lm_head에 투영합니다(동시에는 chunk 하나)."
    )


def _plan_grpo(lengths: LengthTable, resolved: ResolvedConfig, seed: int) -> BatchPlan:
    grpo = resolved.grpo
    if grpo is None:
        raise EstimatorError(
            _issue(ErrorCode.INVALID_REQUEST, Severity.ERROR, "GRPO 설정이 해석되지 않았습니다.")
        )
    layout = resolve_layout(grpo)
    budgets = sorted(set(grpo.completion_budgets))
    if not budgets:
        raise EstimatorError(
            _issue(
                ErrorCode.GRPO_BUDGET_UNSPECIFIED,
                Severity.ERROR,
                "GRPO completion budget이 없어 생성 길이 시나리오를 만들 수 없습니다.",
            )
        )
    prompts = lengths.prompt_tokens
    rows = sum(1 for v in prompts if v is not None)
    if rows == 0:
        raise _no_lengths()
    top = _longest(prompts, 1)[0]
    p_max = prompts[top] or 0
    m = resolved.pad_to_multiple_of
    width_p = round_up(p_max, m)
    b = layout.update_microbatch
    scenarios = []
    for budget in budgets:
        width_l = round_up(budget, m)
        scenarios.append(
            BatchShape(
                name=f"budget_{budget}",
                objective=Objective.GRPO,
                rows_per_microbatch=b,
                sequences_per_forward=b,
                padded_length=width_p + width_l,
                token_slots=b * (width_p + width_l),
                logits_positions=b * (width_l + 1),
                prompt_length=width_p,
                completion_length=width_l,
                source_row_ids=[lengths.row_ids[top]],
                description=(
                    f"GRPO update microbatch: completion {b}개, generation batch 최대 prompt "
                    f"{width_p} + completion budget {width_l} = {width_p + width_l} 토큰 폭"
                    f"(prompt 왼쪽·completion 오른쪽 padding). logits는 sequence마다 "
                    f"{width_l + 1} 위치(logits_to_keep + 1)입니다."
                ),
            )
        )
    worst = scenarios[-1].model_copy(update={"name": "worst_case"})
    dropped = layout.dropped_rows(rows)
    issues = _grpo_issues(layout, resolved, rows, dropped)
    if layout.unique_prompts == 1 or rows == layout.unique_prompts:
        sampler_max: BatchShape | None = _as_sampler_max(
            worst,
            "generation batch마다 prompt가 1개이거나 전체 prompt가 한 generation batch라 "
            "가장 긴 prompt의 batch가 순서와 관계없이 이 shape가 됩니다.",
        )
    else:
        sampler_max = None
        issues.append(_order_issue())
    sampler = SamplerPlan(
        kind="repeat_sampler",
        seed=seed,
        drop_last=False,
        covers_all_rows=dropped == 0,
        dropped_rows=dropped,
        note=_grpo_note(layout, rows, dropped),
    )
    plan_grpo = GrpoBatchPlan(
        unique_prompts_per_generation=layout.unique_prompts,
        num_generations=layout.num_generations,
        live_sequences=layout.live_sequences,
        update_microbatch=b,
        accumulation=layout.accumulation,
        generation_batch_size=layout.generation_batch_size,
        steps_per_generation=layout.steps_per_generation,
        num_iterations=layout.num_iterations,
        completion_budgets=budgets,
        max_prompt_length=p_max,
    )
    return BatchPlan(
        objective=Objective.GRPO,
        unit="completions",
        microbatch=b,
        accumulation=layout.accumulation,
        effective_batch=b * layout.accumulation,
        pad_to_multiple_of=m,
        padding_side="left",
        sampler=sampler,
        worst_case=worst,
        sampler_max=sampler_max,
        scenarios=scenarios,
        grpo=plan_grpo,
        issues=issues,
        batch_key=_batch_key(
            lengths, resolved, seed, "grpo_left_prompt_right_completion", sampler, b
        ),
    )


def _grpo_note(layout: GrpoLayout, rows: int, dropped: int) -> str:
    base = (
        f"TRL RepeatSampler: generation마다 U = generation_batch_size / num_generations = "
        f"{layout.unique_prompts}개 prompt를 {layout.num_generations}번씩 생성합니다."
    )
    if dropped:
        return f"{base} N({rows:,}) mod U = {dropped:,}개 prompt가 epoch마다 무작위로 빠집니다."
    return f"{base} N({rows:,})이 U의 배수라 epoch마다 모든 prompt를 사용합니다."


def _grpo_issues(
    layout: GrpoLayout, resolved: ResolvedConfig, rows: int, dropped: int
) -> list[Issue]:
    issues: list[Issue] = []
    if dropped:
        issues.append(
            _issue(
                ErrorCode.SAMPLER_DROPS_ROWS,
                Severity.ERROR,
                f"GRPO RepeatSampler가 epoch마다 {dropped:,}개 prompt(N mod U, U="
                f"{layout.unique_prompts})를 무작위로 제외합니다. 같은 row가 다음 epoch에도 빠질 "
                "수 있습니다. 모든 row를 쓰려면 U(= generation_batch_size / num_generations)가 "
                f"전체 prompt 수({rows:,})의 약수가 되도록 설정해야 하며, 중복 샘플을 채워 넣는 "
                "방법은 학습 분포를 바꿉니다.",
                rows=rows,
                unique_prompts_per_generation=layout.unique_prompts,
                dropped_rows=dropped,
            )
        )
    grpo = resolved.grpo
    if grpo is not None and grpo.live_sequences != layout.live_sequences:
        issues.append(
            _issue(
                ErrorCode.CONFLICTING_OPTIONS,
                Severity.WARNING,
                f"동시 생성 수가 TRL 관계식(per_device_train_batch_size × steps_per_generation = "
                f"{layout.live_sequences})과 다릅니다({grpo.live_sequences}). 계획은 TRL 관계식을 "
                "따릅니다.",
                resolved_live_sequences=grpo.live_sequences,
                trl_live_sequences=layout.live_sequences,
            )
        )
    if (resolved.microbatch, resolved.accumulation) != (
        layout.update_microbatch,
        layout.accumulation,
    ):
        issues.append(
            _issue(
                ErrorCode.CONFLICTING_OPTIONS,
                Severity.WARNING,
                f"공통 microbatch/accumulation({resolved.microbatch}/{resolved.accumulation})이 "
                f"GRPO 설정({layout.update_microbatch}/{layout.accumulation})과 다릅니다. 계획은 "
                "GRPO 설정(per_device_train_batch_size, gradient_accumulation_steps)을 따릅니다.",
                resolved_microbatch=resolved.microbatch,
                resolved_accumulation=resolved.accumulation,
                grpo_update_microbatch=layout.update_microbatch,
                grpo_accumulation=layout.accumulation,
            )
        )
    return issues


def _batch_key(
    lengths: LengthTable,
    resolved: ResolvedConfig,
    seed: int,
    collator: str,
    sampler: SamplerPlan,
    microbatch: int,
) -> str:
    grpo = resolved.grpo.model_dump(mode="json") if resolved.grpo else None
    return keys.batch_key(
        preprocess_key=_lengths_digest(lengths),
        collator={
            "name": collator,
            "objective": resolved.objective.value,
            "padding_free": False,
            "loss_path": resolved.loss_path,
            "trl": "1.14.1",
            "planner_version": PLANNER_VERSION,
        },
        sampler={"kind": sampler.kind, "seed": seed, "drop_last": False, "grpo": grpo},
        microbatch=microbatch,
        pad_to_multiple_of=resolved.pad_to_multiple_of,
        packing=False,
        distribution={"num_devices": 1},
    )


__all__ = ["PLANNER_VERSION", "plan_batches"]
