"""DPO preprocessing identical to TRL 1.14.1 ``DPOTrainer._prepare_dataset`` with
``max_length=None`` (trl dpo_trainer.py:1007-1108; docs/research/trl-sft-dpo.md §7-8).

prompt, prompt+chosen and prompt+rejected are each rendered as whole conversations; the
completions are the token suffixes after ``len(prompt_ids)``. The collator concatenates
``prompt_ids + chosen_ids`` / ``prompt_ids + rejected_ids`` (dpo_trainer.py:146-170), so the
lengths the model sees are ``len(prompt_ids) + len(<branch>_ids)`` — stored as the branch totals.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from vramforge_estimator.inspection import TokenizerHandle
from vramforge_estimator.schemas import (
    ColumnMapping,
    DatasetFormat,
    EmptySystemPolicy,
    ErrorCode,
    Objective,
)

from .base import TokenizedRecord
from .mapping import Message, RowError, content_digest
from .trl_common import TRL_VERSION, TrlAdapterBase, mapping_error


def extract_prompt(
    chosen: list[Message], rejected: list[Message]
) -> tuple[list[Message], list[Message], list[Message]]:
    """TRL ``extract_prompt`` (trl data_utils.py:557-641) for message lists, quirks included:
    when one list is a prefix of the other the last shared turn stays in the completions."""
    n = min(len(chosen), len(rejected))
    if n == 0:
        # TRL fails here (unbound loop index); report the row instead of guessing.
        raise RowError(
            ErrorCode.DATASET_FORMAT_UNSUPPORTED, "chosen/rejected 대화가 비어 있습니다."
        )
    idx = 0
    for idx in range(n):
        if chosen[idx] != rejected[idx]:
            break
    return chosen[:idx], chosen[idx:], rejected[idx:]


class TrlDpoAdapter(TrlAdapterBase):
    name: ClassVar[str] = f"trl-{TRL_VERSION}-dpo"
    version: ClassVar[str] = "1"
    objective: ClassVar[Objective] = Objective.DPO

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
        if mapping.format not in (DatasetFormat.AUTO, DatasetFormat.PREFERENCE):
            raise mapping_error(
                "DPO에는 선호 쌍(preference) 형식의 매핑이 필요합니다.", format=mapping.format.value
            )
        if not mapping.chosen or not mapping.rejected:
            raise mapping_error("DPO에는 chosen과 rejected 컬럼 매핑이 모두 필요합니다.")
        self.implicit_prompt = mapping.prompt is None
        if not self.conversational:
            self.require_plain_eos()

    def transformation_note(self) -> str:
        note = (
            "DPO: prompt+chosen과 prompt+rejected를 각각 전체 대화로 렌더링해 두 branch 길이를 "
            "따로 계산합니다. 두 응답을 하나로 합치지 않습니다."
        )
        if self.implicit_prompt:
            note += (
                " prompt 컬럼이 없어 chosen/rejected 대화의 공통 앞부분을 prompt로 사용합니다"
                "(TRL extract_prompt)."
            )
        if not self.conversational:
            note += (
                " chat template이 없어 문자열을 그대로 이어 토큰화하고 응답 끝에 EOS를 붙입니다."
            )
        return note + self.policy_note() + " 매핑하지 않은 컬럼(메타데이터)은 렌더링하지 않습니다."

    def _process(self, row: Mapping[str, Any], row_id: str) -> TokenizedRecord:
        if self.conversational:
            return self._chat(row, row_id)
        return self._plain(row, row_id)

    def _chat(self, row: Mapping[str, Any], row_id: str) -> TokenizedRecord:
        system, omitted = self.mapper.system_messages(row)
        if self.implicit_prompt:
            chosen_full = self.message_list(row, self.mapping.chosen, "chosen")
            rejected_full = self.message_list(row, self.mapping.rejected, "rejected")
            chosen_full, d1 = self.mapper.apply_policy(chosen_full)
            rejected_full, d2 = self.mapper.apply_policy(rejected_full)
            prompt, chosen, rejected = extract_prompt(chosen_full, rejected_full)
            prompt = system + prompt
            omitted += d1 + d2
            if not prompt:
                raise RowError(
                    ErrorCode.DATASET_FORMAT_UNSUPPORTED,
                    "chosen/rejected 대화에서 공통 prompt를 찾을 수 없습니다.",
                )
        else:
            prompt = system + self.mapper.messages(row, self.mapping.prompt, "prompt", "user")
            chosen = self.mapper.messages(row, self.mapping.chosen, "chosen", "assistant")
            rejected = self.mapper.messages(row, self.mapping.rejected, "rejected", "assistant")
            prompt, d0 = self.mapper.apply_policy(prompt)
            chosen, d1 = self.mapper.apply_policy(chosen)
            rejected, d2 = self.mapper.apply_policy(rejected)
            omitted += d0 + d1 + d2
        digest = content_digest({"prompt": prompt, "chosen": chosen, "rejected": rejected})
        prompt_text = self.render(prompt, add_generation_prompt=True)
        chosen_text = self.render(prompt + chosen, add_generation_prompt=False)
        rejected_text = self.render(prompt + rejected, add_generation_prompt=False)
        prompt_ids = self.encode_rendered(prompt_text)
        chosen_ids = self.encode_rendered(chosen_text)
        rejected_ids = self.encode_rendered(rejected_text)
        preserved, issue = self.check_preserved(
            [
                (prompt, prompt_text),
                (prompt + chosen, chosen_text),
                (prompt + rejected, rejected_text),
            ]
        )
        return self._record(
            row_id,
            digest,
            prompt_ids,
            chosen_ids,
            rejected_ids,
            preserved=preserved,
            issue=issue,
            omitted=omitted,
        )

    def _plain(self, row: Mapping[str, Any], row_id: str) -> TokenizedRecord:
        system, omitted = self.mapper.system_messages(row)
        if system:
            self.require_template("system 메시지")
        if self.implicit_prompt:
            raise RowError(
                ErrorCode.COLUMN_MAPPING_REQUIRED,
                "prompt 컬럼 없이 문자열 chosen/rejected를 처리할 수 없습니다. "
                "prompt 컬럼을 매핑하세요.",
            )
        prompt = self.plain_text(row, self.mapping.prompt, "prompt")
        chosen = self.plain_text(row, self.mapping.chosen, "chosen")
        rejected = self.plain_text(row, self.mapping.rejected, "rejected")
        digest = content_digest({"prompt": prompt, "chosen": chosen, "rejected": rejected})
        prompt_ids = self.encode_plain(prompt)
        chosen_ids = self.encode_plain(prompt + self.add_eos(chosen))
        rejected_ids = self.encode_plain(prompt + self.add_eos(rejected))
        record = self._record(
            row_id,
            digest,
            prompt_ids,
            chosen_ids,
            rejected_ids,
            preserved=None,
            issue=None,
            omitted=omitted,
        )
        for ids in (chosen_ids, rejected_ids):
            record.extras.update(self.special_token_flags(ids))
        return record

    def _record(
        self,
        row_id: str,
        digest: str,
        prompt_ids: list[int],
        prompt_chosen_ids: list[int],
        prompt_rejected_ids: list[int],
        *,
        preserved: bool | None,
        issue: str | None,
        omitted: int,
    ) -> TokenizedRecord:
        p = len(prompt_ids)
        chosen = len(prompt_chosen_ids[p:])
        rejected = len(prompt_rejected_ids[p:])
        extras: dict[str, Any] = {}
        if omitted:
            extras["system_omitted"] = omitted
        if prompt_chosen_ids[:p] != prompt_ids or prompt_rejected_ids[:p] != prompt_ids:
            extras["prefix_mismatch"] = True  # TRL only warns and slices anyway
        return TokenizedRecord(
            row_id=row_id,
            objective=self.objective,
            ok=True,
            prompt_tokens=p,
            chosen_total_tokens=p + chosen,
            rejected_total_tokens=p + rejected,
            chosen_completion_tokens=chosen,
            rejected_completion_tokens=rejected,
            content_digest=digest,
            template_preserved=preserved,
            template_issue=issue,
            extras=extras,
        )


__all__ = ["TrlDpoAdapter", "extract_prompt"]
