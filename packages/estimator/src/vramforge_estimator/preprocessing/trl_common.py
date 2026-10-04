"""Shared machinery of the TRL 1.14.1 preprocessing adapters (plan.md §7.3).

Tokenizer calls mirror what TRL 1.14.1 does through transformers 5.18 (no torch):

* conversational input: ``apply_chat_template(..., tokenize=True)`` renders the template and then
  tokenizes with ``add_special_tokens=False, truncation=False`` (transformers
  tokenization_utils_base.py:3123-3132). We render once with ``tokenize=False`` and run the same
  tokenizer call, which yields identical ids and gives us the rendered text for the content check.
* plain strings: ``processing_class(text=...)`` with the tokenizer defaults (trl data_utils.py
  ``_tokenize``); no max_length is ever passed, so nothing is truncated.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from vramforge_estimator.errors import EstimatorError, make_issue
from vramforge_estimator.inspection import TokenizerHandle
from vramforge_estimator.schemas import (
    ColumnMapping,
    EmptySystemPolicy,
    ErrorCode,
    Objective,
    Stage,
)

from .base import TokenizedRecord
from .mapping import (
    RESERVED_TEMPLATE_KEYS,
    TRUNCATION_KEYS,
    Message,
    RecordMapper,
    RowError,
    text_parts,
)

TRL_VERSION = "1.14.1"
_MIN_LEAK_CHECK_CHARS = 4


def check_template_kwargs(template_kwargs: Mapping[str, Any] | None) -> dict[str, Any]:
    """Template kwargs may only carry template variables (e.g. ``enable_thinking``)."""
    kwargs = dict(template_kwargs or {})
    truncating = sorted(set(kwargs) & TRUNCATION_KEYS)
    if truncating:
        raise EstimatorError(
            make_issue(
                ErrorCode.DATA_PRESERVATION_VIOLATION,
                "chat template 옵션에 길이 절단 인자(truncation/max_length 등)를 넣을 수 없습니다. "
                "무절단 분석에서는 템플릿 변수만 허용합니다.",
                stage=Stage.TOKENIZING,
                keys=truncating,
            )
        )
    reserved = sorted(set(kwargs) & RESERVED_TEMPLATE_KEYS)
    if reserved:
        raise EstimatorError(
            make_issue(
                ErrorCode.CONFLICTING_OPTIONS,
                "chat template 옵션에는 템플릿 변수만 넣을 수 있습니다. 토큰화 인자는 지정할 수 "
                "없습니다.",
                stage=Stage.TOKENIZING,
                keys=reserved,
            )
        )
    return kwargs


def mapping_error(message: str, **details: object) -> EstimatorError:
    return EstimatorError(
        make_issue(ErrorCode.COLUMN_MAPPING_REQUIRED, message, stage=Stage.TOKENIZING, **details)
    )


def _safe_reason(exc: BaseException, texts: Sequence[str]) -> str:
    """Exception type, plus the template's own message when it carries no row text."""
    name = type(exc).__name__
    try:
        from jinja2 import TemplateError
    except ImportError:  # pragma: no cover - jinja2 ships with the analysis extra
        return name
    if isinstance(exc, TemplateError):
        detail = str(exc).strip()
        leaks = any(len(t) >= _MIN_LEAK_CHECK_CHARS and t in detail for t in texts)
        if detail and len(detail) <= 300 and not leaks:
            return f"{name}: {detail}"
    return name


class TrlAdapterBase:
    """Common state and tokenizer helpers. Subclasses implement `_process` and the note."""

    name: ClassVar[str]
    version: ClassVar[str]
    objective: ClassVar[Objective]

    def __init__(
        self,
        handle: TokenizerHandle,
        mapping: ColumnMapping,
        *,
        template_kwargs: Mapping[str, Any] | None = None,
        empty_system_policy: EmptySystemPolicy = EmptySystemPolicy.OMIT,
    ) -> None:
        self.handle = handle
        self.tokenizer = handle.tokenizer
        self.mapping = mapping
        self.empty_system_policy = empty_system_policy
        self.template_kwargs = check_template_kwargs(template_kwargs)
        self.mapper = RecordMapper(mapping, empty_system_policy)
        # TRL decides "conversational" from the data; our mapping renders string columns as chat
        # messages whenever the model ships a chat template (plan §7.2), else as plain strings.
        self.conversational = bool(getattr(self.tokenizer, "chat_template", None))
        self.eos_token: str | None = getattr(self.tokenizer, "eos_token", None)
        self._bos_id: int | None = getattr(self.tokenizer, "bos_token_id", None)
        self._eos_id: int | None = getattr(self.tokenizer, "eos_token_id", None)

    # ------------------------------------------------------------------ contract

    def transformation_note(self) -> str:
        raise NotImplementedError

    def process(self, row: Mapping[str, Any], row_id: str) -> TokenizedRecord:
        """Tokenize one row without truncation. Failures are returned, never raised."""
        try:
            return self._process(row, row_id)
        except RowError as exc:
            return self.failed(row_id, exc.code, exc.message)
        except Exception as exc:  # safety net: one bad row must not stop the scan
            return self.failed(
                row_id,
                ErrorCode.INTERNAL_ERROR,
                f"전처리 중 예기치 않은 오류가 발생했습니다 ({type(exc).__name__}).",
            )

    def _process(self, row: Mapping[str, Any], row_id: str) -> TokenizedRecord:
        raise NotImplementedError

    # ------------------------------------------------------------------ helpers

    def failed(
        self, row_id: str, code: ErrorCode, message: str, digest: str = ""
    ) -> TokenizedRecord:
        return TokenizedRecord(
            row_id=row_id,
            objective=self.objective,
            ok=False,
            content_digest=digest,
            error_code=code,
            error_message=message,
        )

    def require_template(self, what: str) -> None:
        if not self.conversational:
            raise RowError(
                ErrorCode.TEMPLATE_REQUIRED,
                f"모델에 chat template이 없어 {what}을(를) 렌더링할 수 없습니다.",
            )

    def plain_text(self, row: Mapping[str, Any], column: str | None, field: str) -> str:
        """A string value for the non-conversational path (lists need a chat template)."""
        value = self.mapper.value(row, column, field)
        if isinstance(value, list | tuple):
            self.require_template("메시지 목록")
        if not isinstance(value, str):
            raise RowError(
                ErrorCode.DATASET_FORMAT_UNSUPPORTED,
                f"'{field}' 값은 문자열이어야 합니다 ({type(value).__name__}).",
            )
        return value

    def message_list(self, row: Mapping[str, Any], column: str | None, field: str) -> list[Message]:
        """A column that must already hold a list of chat messages."""
        value = self.mapper.value(row, column, field)
        if not isinstance(value, list | tuple):
            raise RowError(
                ErrorCode.DATASET_FORMAT_UNSUPPORTED,
                f"'{field}' 값은 메시지 목록이어야 합니다 ({type(value).__name__}).",
            )
        return self.mapper.messages(row, column, field, "user")

    def render(self, messages: list[Message], *, add_generation_prompt: bool) -> str:
        if not messages:
            raise RowError(
                ErrorCode.DATASET_FORMAT_UNSUPPORTED, "렌더링할 메시지가 없습니다(빈 대화)."
            )
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                **self.template_kwargs,
            )
        except Exception as exc:
            texts = [t for m in messages for t in text_parts(m.get("content"))]
            raise RowError(
                ErrorCode.DATASET_FORMAT_UNSUPPORTED,
                f"chat template 렌더링에 실패했습니다 ({_safe_reason(exc, texts)}).",
            ) from exc
        if not isinstance(text, str):  # pragma: no cover - defensive
            raise RowError(ErrorCode.INTERNAL_ERROR, "chat template 결과가 문자열이 아닙니다.")
        return text

    def encode_rendered(self, text: str) -> list[int]:
        """Same call apply_chat_template(tokenize=True) makes after rendering."""
        try:
            ids = self.tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
        except Exception as exc:
            raise RowError(
                ErrorCode.DATASET_FORMAT_UNSUPPORTED,
                f"토큰화에 실패했습니다 ({type(exc).__name__}).",
            ) from exc
        return list(ids)

    def encode_plain(self, text: str) -> list[int]:
        """TRL's non-conversational path: processing_class(text=...) with tokenizer defaults."""
        try:
            ids = self.tokenizer(text=text, truncation=False)["input_ids"]
        except Exception as exc:
            raise RowError(
                ErrorCode.DATASET_FORMAT_UNSUPPORTED,
                f"토큰화에 실패했습니다 ({type(exc).__name__}).",
            ) from exc
        return list(ids)

    def encode_with_assistant_mask(self, messages: list[Message]) -> tuple[list[int], list[int]]:
        """apply_chat_template(return_assistant_tokens_mask=True) for assistant-only loss."""
        try:
            out = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_assistant_tokens_mask=True,
                **self.template_kwargs,
            )
        except Exception as exc:
            texts = [t for m in messages for t in text_parts(m.get("content"))]
            raise RowError(
                ErrorCode.DATASET_FORMAT_UNSUPPORTED,
                f"chat template 렌더링에 실패했습니다 ({_safe_reason(exc, texts)}).",
            ) from exc
        return list(out["input_ids"]), list(out["assistant_masks"])

    def add_eos(self, text: str) -> str:
        """TRL appends the EOS string to non-conversational text/completions that lack it."""
        assert self.eos_token is not None  # checked at construction for plain layouts
        return text if text.endswith(self.eos_token) else text + self.eos_token

    @staticmethod
    def check_preserved(rendered: Sequence[tuple[list[Message], str]]) -> tuple[bool, str | None]:
        """Every mapped message content must appear verbatim in its rendered text (plan §7.4)."""
        missing: dict[str, int] = {}
        for messages, text in rendered:
            for message in messages:
                for piece in text_parts(message.get("content")):
                    if piece not in text:
                        role = str(message.get("role"))
                        missing[role] = missing.get(role, 0) + 1
        if not missing:
            return True, None
        parts = ", ".join(f"{role} {count}개" for role, count in sorted(missing.items()))
        return (
            False,
            f"chat template 렌더링 결과에 원문 그대로 포함되지 않은 content가 있습니다 ({parts}).",
        )

    def special_token_flags(self, ids: Sequence[int]) -> dict[str, Any]:
        """Flags duplicated BOS/EOS (e.g. BOS text in a string the tokenizer also prefixes)."""
        flags: dict[str, Any] = {}
        if self._bos_id is not None and len(ids) >= 2 and ids[0] == ids[1] == self._bos_id:
            flags["duplicate_bos"] = True
        if self._eos_id is not None and len(ids) >= 2 and ids[-1] == ids[-2] == self._eos_id:
            flags["duplicate_eos"] = True
        return flags

    def require_plain_eos(self) -> None:
        if self.eos_token is None:
            raise EstimatorError(
                make_issue(
                    ErrorCode.TOKENIZER_REQUIRED,
                    "EOS token이 없는 tokenizer로는 TRL의 비대화형(plain text) 전처리를 재현할 수 "
                    "없습니다.",
                    stage=Stage.TOKENIZING,
                )
            )

    def policy_note(self) -> str:
        if self.mapping.system is None:
            return ""
        if self.empty_system_policy is EmptySystemPolicy.OMIT:
            return " 빈 system 값(공백 포함)은 system 메시지로 렌더링하지 않습니다(정책: omit)."
        return " 빈 system 값도 system 메시지로 렌더링합니다(정책: keep)."


def loss_positions(masks: Sequence[Sequence[int]], length: int) -> int:
    """Positions that receive loss: labels[1:] != -100 after the causal shift, where a label is
    kept only if every applicable mask is 1 (trl sft_trainer.py build_labels, :1627-1654)."""
    if length <= 1:
        return 0
    if not masks:
        return length - 1
    count = 0
    for i in range(1, length):
        if all(i < len(m) and m[i] for m in masks):
            count += 1
    return count


__all__ = [
    "TRL_VERSION",
    "TrlAdapterBase",
    "check_template_kwargs",
    "loss_positions",
    "mapping_error",
]
