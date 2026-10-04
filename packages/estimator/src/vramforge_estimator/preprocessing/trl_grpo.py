"""GRPO prompt tokenization identical to TRL 1.14.1 ``GRPOTrainer._tokenize_prompts``
(trl grpo_trainer.py:1769-1846; docs/research/trl-grpo.md §4.1, §9, R6).

Conversational prompts: ``apply_chat_template(prompt, add_generation_prompt=True,
tokenize=True)`` (no special tokens added, no truncation). TRL 1.14.1 has no
``max_prompt_length``; template kwargs are forwarded as-is, which is why truncation keys are
rejected (trl_common.check_template_kwargs). Plain prompts: ``processing_class(text=...)``.
Completion lengths are not data: they are budget scenarios planned later (plan §8.3).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vramforge_estimator.inspection import TokenizerHandle
from vramforge_estimator.schemas import ColumnMapping, DatasetFormat, EmptySystemPolicy, Objective

from .base import TokenizedRecord
from .mapping import content_digest
from .trl_common import TRL_VERSION, TrlAdapterBase, mapping_error


class TrlGrpoAdapter(TrlAdapterBase):
    name = f"trl-{TRL_VERSION}-grpo"
    version = "1"
    objective = Objective.GRPO

    def __init__(
        self,
        handle: TokenizerHandle,
        mapping: ColumnMapping,
        *,
        template_kwargs: Mapping[str, Any] | None = None,
        empty_system_policy: EmptySystemPolicy = EmptySystemPolicy.OMIT,
    ) -> None:
        super().__init__(
            handle,
            mapping,
            template_kwargs=template_kwargs,
            empty_system_policy=empty_system_policy,
        )
        if mapping.format in (DatasetFormat.MESSAGES, DatasetFormat.TEXT) or not mapping.prompt:
            raise mapping_error(
                "GRPO에는 prompt 컬럼 매핑이 필요합니다.", format=mapping.format.value
            )

    def transformation_note(self) -> str:
        note = (
            "GRPO: prompt만 사용합니다(생성 prompt 포함). chosen/rejected 응답은 생성 길이나 "
            "보상 계산에 사용하지 않으며, 응답 길이는 completion budget 시나리오로 계산합니다."
        )
        if not self.conversational:
            note += " chat template이 없어 prompt 문자열을 그대로 토큰화합니다."
        return note + self.policy_note() + " 매핑하지 않은 컬럼(메타데이터)은 렌더링하지 않습니다."

    def _process(self, row: Mapping[str, Any], row_id: str) -> TokenizedRecord:
        system, omitted = self.mapper.system_messages(row)
        extras: dict[str, Any] = {}
        if self.conversational:
            prompt = system + self.mapper.messages(row, self.mapping.prompt, "prompt", "user")
            prompt, dropped = self.mapper.apply_policy(prompt)
            omitted += dropped
            digest = content_digest({"prompt": prompt})
            text = self.render(prompt, add_generation_prompt=True)
            ids = self.encode_rendered(text)
            preserved, issue = self.check_preserved([(prompt, text)])
        else:
            if system:
                self.require_template("system 메시지")
            plain = self.plain_text(row, self.mapping.prompt, "prompt")
            digest = content_digest({"prompt": plain})
            ids = self.encode_plain(plain)
            preserved, issue = None, None
            extras.update(self.special_token_flags(ids))
        if omitted:
            extras["system_omitted"] = omitted
        return TokenizedRecord(
            row_id=row_id,
            objective=self.objective,
            ok=True,
            prompt_tokens=len(ids),
            content_digest=digest,
            template_preserved=preserved,
            template_issue=issue,
            extras=extras,
        )


__all__ = ["TrlGrpoAdapter"]
