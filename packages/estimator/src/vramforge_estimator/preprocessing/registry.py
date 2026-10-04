"""Preprocessing adapter registry: one TRL 1.14.1 adapter per objective (plan.md §7.3, §16.3).

The adapter's ``name`` and ``version`` are part of ``preprocess_key``; bump ``version`` in the
adapter module whenever the reproduced behavior changes.
"""

from __future__ import annotations

from typing import Any

from vramforge_estimator.errors import EstimatorError, make_issue
from vramforge_estimator.inspection import TokenizerHandle
from vramforge_estimator.schemas import (
    ColumnMapping,
    EmptySystemPolicy,
    ErrorCode,
    Objective,
    Stage,
)

from .base import PreprocessingAdapter
from .trl_dpo import TrlDpoAdapter
from .trl_grpo import TrlGrpoAdapter
from .trl_sft import TrlSftAdapter

_ADAPTERS: dict[Objective, type[TrlSftAdapter] | type[TrlDpoAdapter] | type[TrlGrpoAdapter]] = {
    Objective.SFT: TrlSftAdapter,
    Objective.DPO: TrlDpoAdapter,
    Objective.GRPO: TrlGrpoAdapter,
}


def get_adapter(
    objective: Objective,
    tokenizer: TokenizerHandle,
    mapping: ColumnMapping,
    *,
    template_kwargs: dict[str, Any] | None = None,
    empty_system_policy: EmptySystemPolicy = EmptySystemPolicy.OMIT,
) -> PreprocessingAdapter:
    """Return the adapter for `objective`; raises `EstimatorError` (COLUMN_MAPPING_REQUIRED)
    when the mapping cannot produce the records the objective needs.

    `empty_system_policy` must equal `mapping.empty_system_policy`: the mapping is what goes into
    ``preprocess_key``, so a silent override would let two different tokenizations share a cache
    entry. Template kwargs may only carry template variables (never truncation arguments).
    """
    if mapping.empty_system_policy is not empty_system_policy:
        raise EstimatorError(
            make_issue(
                ErrorCode.CONFLICTING_OPTIONS,
                "빈 system 처리 정책이 컬럼 매핑과 요청 설정에서 서로 다릅니다.",
                stage=Stage.TOKENIZING,
                mapping_policy=mapping.empty_system_policy.value,
                requested_policy=empty_system_policy.value,
            )
        )
    adapter_cls = _ADAPTERS.get(objective)
    if adapter_cls is None:  # pragma: no cover - Objective is a closed enum
        raise EstimatorError(
            make_issue(
                ErrorCode.UNSUPPORTED_BACKEND_COMBINATION,
                "지원하지 않는 학습 방식입니다.",
                stage=Stage.TOKENIZING,
                objective=str(objective),
            )
        )
    return adapter_cls(
        tokenizer,
        mapping,
        template_kwargs=template_kwargs,
        empty_system_policy=empty_system_policy,
    )
