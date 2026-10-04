"""SFT preprocessing identical to TRL 1.14.1 ``SFTTrainer._prepare_dataset`` with
``max_length=None`` and ``packing=False`` (trl sft_trainer.py:1460-1710;
docs/research/trl-sft-dpo.md §2).

Layouts (TRL dataset types):

* ``prompt_completion`` — prompt=[system?, user], completion=[assistant]. Prompt ids come from
  ``apply_chat_template(prompt, add_generation_prompt=True)``, the sequence from rendering
  prompt+completion once; ``completion_mask = [0]*len(prompt) + [1]*rest`` and, because
  ``completion_only_loss`` defaults to True for this type, prompt tokens get no loss.
* ``messages`` — the whole conversation rendered once; every token is a label (assistant-only
  loss is optional and needs ``{% generation %}`` markers).
* ``text`` / plain prompt-completion — non-conversational strings: EOS string appended when
  missing, tokenized with the tokenizer defaults (``add_special_tokens=True``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from vramforge_estimator.errors import EstimatorError, make_issue
from vramforge_estimator.inspection import TokenizerHandle
from vramforge_estimator.schemas import (
    ColumnMapping,
    DatasetFormat,
    EmptySystemPolicy,
    ErrorCode,
    Objective,
    Stage,
)

from .base import TokenizedRecord
from .mapping import RowError, content_digest
from .trl_common import TRL_VERSION, TrlAdapterBase, loss_positions, mapping_error, token_digest

SftLayout = Literal["prompt_completion", "messages", "text"]


def resolve_sft_layout(mapping: ColumnMapping) -> tuple[SftLayout, str | None]:
    """Pick the TRL dataset type and the completion column (`completion`, else `chosen`)."""
    fmt = mapping.format
    if fmt is DatasetFormat.PROMPT_ONLY:
        raise mapping_error("SFT에는 응답(completion/chosen) 컬럼이 필요합니다.", format=fmt.value)
    if fmt is DatasetFormat.MESSAGES or (fmt is DatasetFormat.AUTO and mapping.messages):
        if not mapping.messages:
            raise mapping_error("messages 형식에는 messages 컬럼 매핑이 필요합니다.")
        if mapping.prompt or mapping.completion or mapping.chosen:
            raise mapping_error("messages 컬럼과 prompt/응답 컬럼을 동시에 매핑할 수 없습니다.")
        return "messages", None
    if fmt is DatasetFormat.TEXT or (fmt is DatasetFormat.AUTO and mapping.text):
        if not mapping.text:
            raise mapping_error("text 형식에는 text 컬럼 매핑이 필요합니다.")
        if mapping.system:
            raise mapping_error("text 형식에는 system 컬럼을 매핑할 수 없습니다.")
        return "text", None
    if fmt is DatasetFormat.PREFERENCE:
        completion = mapping.chosen
    elif fmt is DatasetFormat.PROMPT_COMPLETION:
        completion = mapping.completion
    else:
        completion = mapping.completion or mapping.chosen
    if not mapping.prompt or not completion:
        raise mapping_error(
            "SFT에는 prompt와 응답(completion 또는 chosen) 컬럼 매핑이 필요합니다.",
            format=fmt.value,
        )
    return "prompt_completion", completion


class TrlSftAdapter(TrlAdapterBase):
    name = f"trl-{TRL_VERSION}-sft"
    version = "2"  # 2: records carry token digests and literal special-token findings
    objective = Objective.SFT

    def __init__(
        self,
        handle: TokenizerHandle,
        mapping: ColumnMapping,
        *,
        template_kwargs: Mapping[str, Any] | None = None,
        empty_system_policy: EmptySystemPolicy = EmptySystemPolicy.OMIT,
        assistant_only_loss: bool = False,
    ) -> None:
        super().__init__(
            handle,
            mapping,
            template_kwargs=template_kwargs,
            empty_system_policy=empty_system_policy,
        )
        self.layout, self.completion_column = resolve_sft_layout(mapping)
        self.assistant_only_loss = assistant_only_loss
        self.uses_chosen = self.completion_column is not None and (
            self.completion_column == mapping.chosen and mapping.completion is None
        )
        if self.layout == "messages" and not self.conversational:
            raise EstimatorError(
                make_issue(
                    ErrorCode.TEMPLATE_REQUIRED,
                    "messages 형식을 렌더링하려면 모델의 chat template이 필요합니다.",
                    stage=Stage.TOKENIZING,
                )
            )
        if assistant_only_loss and (
            self.layout == "text"
            or not self.conversational
            or not handle.manifest.has_generation_markers
        ):
            # TRL swaps in a bundled training template only for exact known templates; that is
            # not reproduced here, so the combination is reported instead of guessed.
            raise EstimatorError(
                make_issue(
                    ErrorCode.TEMPLATE_REQUIRED,
                    "assistant-only loss에는 {% generation %} 마커가 있는 chat template이 "
                    "필요합니다.",
                    stage=Stage.TOKENIZING,
                )
            )
        if self.layout == "text" or not self.conversational:
            self.require_plain_eos()

    def transformation_note(self) -> str:
        if self.layout == "messages":
            note = "SFT: messages 대화 전체를 모델의 chat template으로 한 번에 렌더링해 학습합니다."
        elif self.layout == "text":
            note = (
                "SFT: text 값을 그대로 토큰화하고 끝에 EOS가 없으면 붙입니다(TRL 언어 모델링 형식)."
            )
        elif self.uses_chosen:
            note = (
                "SFT: prompt와 chosen 응답만 학습하고 rejected 응답은 사용하지 않습니다. "
                "이는 선택한 데이터 변환이며 응답을 자르는 절단(truncation)이 아닙니다."
            )
        else:
            note = "SFT: prompt와 completion을 이어 학습하며 loss는 completion 토큰에만 적용합니다."
        if self.layout == "prompt_completion" and not self.conversational:
            note += (
                " chat template이 없어 문자열을 그대로 이어 토큰화하고 응답 끝에 EOS를 붙입니다."
            )
        return note + self.policy_note() + " 매핑하지 않은 컬럼(메타데이터)은 렌더링하지 않습니다."

    # ------------------------------------------------------------------ processing

    def _process(self, row: Mapping[str, Any], row_id: str) -> TokenizedRecord:
        if self.layout == "messages":
            return self._messages(row, row_id)
        if self.layout == "text":
            return self._text(row, row_id)
        if self.conversational:
            return self._prompt_completion_chat(row, row_id)
        return self._prompt_completion_plain(row, row_id)

    def _messages(self, row: Mapping[str, Any], row_id: str) -> TokenizedRecord:
        system, omitted = self.mapper.system_messages(row)
        messages = system + self.message_list(row, self.mapping.messages, "messages")
        messages, dropped = self.mapper.apply_policy(messages)
        omitted += dropped
        example = {"messages": messages}
        digest = content_digest(example)
        text = self.render(messages, add_generation_prompt=False)
        if self.assistant_only_loss:
            ids, assistant_mask = self.encode_with_assistant_mask(messages)
            if 1 not in assistant_mask:
                # TRL raises RuntimeError for the whole dataset in this case.
                raise RowError(
                    ErrorCode.TEMPLATE_REQUIRED,
                    "assistant-only loss인데 assistant 토큰이 없습니다.",
                )
            masks = [assistant_mask]
        else:
            ids, masks = self.encode_rendered(text), []
        preserved, issue = self.check_preserved([(messages, text)])
        return self._record(
            row_id,
            digest,
            ids,
            prompt_len=None,
            masks=masks,
            preserved=preserved,
            issue=issue,
            omitted=omitted,
            content=self.message_texts(messages),
        )

    def _prompt_completion_chat(self, row: Mapping[str, Any], row_id: str) -> TokenizedRecord:
        system, omitted = self.mapper.system_messages(row)
        prompt = system + self.mapper.messages(row, self.mapping.prompt, "prompt", "user")
        completion = self.mapper.messages(row, self.completion_column, "completion", "assistant")
        prompt, dropped_p = self.mapper.apply_policy(prompt)
        completion, dropped_c = self.mapper.apply_policy(completion)
        omitted += dropped_p + dropped_c
        digest = content_digest({"prompt": prompt, "completion": completion})
        prompt_ids = self.encode_rendered(self.render(prompt, add_generation_prompt=True))
        full = prompt + completion
        text = self.render(full, add_generation_prompt=False)
        masks: list[list[int]] = []
        if self.assistant_only_loss:
            ids, assistant_mask = self.encode_with_assistant_mask(full)
            if 1 not in assistant_mask:
                raise RowError(
                    ErrorCode.TEMPLATE_REQUIRED,
                    "assistant-only loss인데 assistant 토큰이 없습니다.",
                )
            masks.append(assistant_mask)
        else:
            ids = self.encode_rendered(text)
        preserved, issue = self.check_preserved([(full, text)])
        return self._record(
            row_id,
            digest,
            ids,
            prompt_len=len(prompt_ids),
            masks=masks,
            preserved=preserved,
            issue=issue,
            omitted=omitted,
            prefix_ok=ids[: len(prompt_ids)] == prompt_ids,
            content=self.message_texts(full),
        )

    def _prompt_completion_plain(self, row: Mapping[str, Any], row_id: str) -> TokenizedRecord:
        system, omitted = self.mapper.system_messages(row)
        if system:
            self.require_template("system 메시지")
        prompt = self.plain_text(row, self.mapping.prompt, "prompt")
        completion = self.plain_text(row, self.completion_column, "completion")
        digest = content_digest({"prompt": prompt, "completion": completion})
        content = [prompt, self.without_final_eos(completion)]
        completion = self.add_eos(completion)
        prompt_ids = self.encode_plain(prompt)
        ids = self.encode_plain(prompt + completion)
        record = self._record(
            row_id,
            digest,
            ids,
            prompt_len=len(prompt_ids),
            masks=[],
            preserved=None,
            issue=None,
            omitted=omitted,
            prefix_ok=ids[: len(prompt_ids)] == prompt_ids,
            content=content,
        )
        record.extras.update(self.special_token_flags(ids))
        return record

    def _text(self, row: Mapping[str, Any], row_id: str) -> TokenizedRecord:
        text = self.plain_text(row, self.mapping.text, "text")
        digest = content_digest({"text": text})
        ids = self.encode_plain(self.add_eos(text))
        record = self._record(
            row_id,
            digest,
            ids,
            prompt_len=None,
            masks=[],
            preserved=None,
            issue=None,
            omitted=0,
            content=[self.without_final_eos(text)],
        )
        record.extras.update(self.special_token_flags(ids))
        return record

    def _record(
        self,
        row_id: str,
        digest: str,
        ids: list[int],
        *,
        prompt_len: int | None,
        masks: list[list[int]],
        preserved: bool | None,
        issue: str | None,
        omitted: int,
        prefix_ok: bool | None = None,
        content: Sequence[str] = (),
    ) -> TokenizedRecord:
        total = len(ids)
        completion: int | None = None
        if prompt_len is not None:
            # completion_mask = [0]*len(prompt_ids) + [1]*(len(ids) - len(prompt_ids)); with
            # completion_only_loss (default for prompt-completion) it gates the labels.
            completion = max(0, total - prompt_len)
            masks = [[0] * prompt_len + [1] * completion, *masks]
        extras: dict[str, Any] = {"token_digest": token_digest({"input_ids": ids})}
        if omitted:
            extras["system_omitted"] = omitted
        if prefix_ok is False:
            extras["prefix_mismatch"] = True
        extras.update(self.literal_token_flags(content))
        return TokenizedRecord(
            row_id=row_id,
            objective=self.objective,
            ok=True,
            prompt_tokens=prompt_len,
            completion_tokens=completion,
            sequence_tokens=total,
            loss_token_count=loss_positions(masks, total),
            content_digest=digest,
            template_preserved=preserved,
            template_issue=issue,
            extras=extras,
        )


__all__ = ["TrlSftAdapter", "resolve_sft_layout"]
