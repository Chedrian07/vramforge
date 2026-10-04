"""Shared machinery of the TRL 1.14.1 preprocessing adapters (plan.md §7.3).

Tokenizer calls mirror what TRL 1.14.1 does through transformers 5.18 (no torch):

* conversational input: ``apply_chat_template(..., tokenize=True)`` renders the template and then
  tokenizes with ``add_special_tokens=False, truncation=False`` (transformers
  tokenization_utils_base.py:3123-3132). We render once with ``tokenize=False`` and run the same
  tokenizer call, which yields identical ids and gives us the rendered text for the content check.
* plain strings: ``processing_class(text=...)`` with the tokenizer defaults (trl data_utils.py
  ``_tokenize``); no max_length is ever passed, so nothing is truncated.

Every successful record carries ``extras["token_digest"]`` (see `token_digest`): a hash of the
token ids the trainer feeds to the model for that row, so GPU validation can confirm that a
re-read row tokenizes identically without the ids ever being stored (plan §7.6). Rows whose mapped
content spells out an added token (e.g. ``<|im_end|>``) get ``extras["special_token_literal"]``:
the tokenizer turns that text into the control token, exactly as in training, so the lengths stay
TRL's and the finding is only a warning (docs/research/example-model-dataset.md R7b).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from vramforge_estimator.errors import EstimatorError, make_issue
from vramforge_estimator.inspection import TokenizerHandle
from vramforge_estimator.schemas import (
    ColumnMapping,
    EmptySystemPolicy,
    ErrorCode,
    Issue,
    Objective,
    Severity,
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
_MAX_REASON_CHARS = 300


def token_digest(sequences: Mapping[str, Sequence[int]]) -> str:
    """sha256 (hex) of the token ids one row feeds to the model, combined over its sequences.

    Encoding, so any tool can recompute it from the trainer's own tensors: for each sequence in
    sorted name order, the ASCII bytes of ``"<name>:" + ",".join(str(i) for i in ids) + "\\n"``.
    Names are the TRL 1.14.1 model inputs of one row: SFT ``input_ids`` (the tokenized dataset
    column); DPO ``chosen_input_ids`` and ``rejected_input_ids`` (``prompt_ids + chosen_ids`` and
    ``prompt_ids + rejected_ids``, exactly as the collator concatenates them); GRPO
    ``prompt_ids`` (``_tokenize_prompts``).
    """
    h = hashlib.sha256()
    for name in sorted(sequences):
        h.update(f"{name}:".encode("ascii"))
        h.update(",".join(map(str, sequences[name])).encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def _markup_like(content: str) -> bool:
    """``<tool_call>``, ``[INST]``: added tokens that act as control markers. Other non-special
    added tokens (e.g. whitespace runs or extra characters some vocabularies add) are ordinary
    text pieces and are not reported."""
    text = content.strip()
    return len(text) >= 3 and (text[0], text[-1]) in (("<", ">"), ("[", "]"))


def literal_token_matcher(tokenizer: Any) -> tuple[dict[int, str], re.Pattern[str] | None]:
    """Added tokens that raw text can turn into (id -> string) and a regex that finds them.

    Special tokens count unless the tokenizer splits them (``split_special_tokens``); non-special
    added tokens count when they look like control markers (``<think>``, ``<tool_call>``)."""
    try:
        decoder = dict(getattr(tokenizer, "added_tokens_decoder", None) or {})
    except Exception:  # pragma: no cover - defensive against unusual tokenizer classes
        return {}, None
    split_special = bool(getattr(tokenizer, "split_special_tokens", False))
    tokens: dict[int, str] = {}
    for index, added in decoder.items():
        content = getattr(added, "content", None)
        if not isinstance(content, str) or not content.strip():
            continue
        special = bool(getattr(added, "special", False))
        if (special and split_special) or (not special and not _markup_like(content)):
            continue
        tokens[int(index)] = content
    if not tokens:
        return {}, None
    literals = sorted(set(tokens.values()), key=len, reverse=True)
    return tokens, re.compile("|".join(re.escape(t) for t in literals))


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
        Issue(
            code=ErrorCode.COLUMN_MAPPING_REQUIRED,
            severity=Severity.ERROR,
            stage=Stage.TOKENIZING,
            user_message=message,
            details=dict(details),
        )
    )


def _template_source(tokenizer: Any) -> str:
    template = getattr(tokenizer, "chat_template", None)
    if isinstance(template, Mapping):  # named templates
        return "\n".join(str(v) for v in template.values())
    return template if isinstance(template, str) else ""


def _safe_reason(exc: BaseException, template_source: str) -> str:
    """Exception type, plus the template's own message only when it is literal template text.

    Templates build messages from row values too (e.g. ``'Unknown role: ' ~ message.role`` or
    tool-call arguments), and failed-row messages end up in results and exports, so anything that
    is not verbatim template text is dropped rather than scanned for leaks.
    """
    name = type(exc).__name__
    try:
        from jinja2 import TemplateError
    except ImportError:  # pragma: no cover - jinja2 ships with the analysis extra
        return name
    if isinstance(exc, TemplateError):
        detail = str(exc).strip()
        if detail and len(detail) <= _MAX_REASON_CHARS and detail in template_source:
            return f"{name}: {detail}"
    return name


class TrlAdapterBase:
    """Common state and tokenizer helpers. Subclasses implement `_process` and the note."""

    name: str
    version: str
    objective: Objective

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
        self._literal_tokens, self._literal_pattern = literal_token_matcher(self.tokenizer)

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
            raise self._template_failure(exc) from exc
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
            raise self._template_failure(exc) from exc
        return list(out["input_ids"]), list(out["assistant_masks"])

    def _template_failure(self, exc: BaseException) -> RowError:
        reason = _safe_reason(exc, _template_source(self.tokenizer))
        return RowError(
            ErrorCode.DATASET_FORMAT_UNSUPPORTED,
            f"chat template 렌더링에 실패했습니다 ({reason}).",
        )

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

    def literal_token_flags(self, pieces: Iterable[str]) -> dict[str, Any]:
        """Added-token strings that mapped content spells out verbatim and that the tokenizer
        turns into that token. A regex pre-filter keeps clean rows cheap; a hit is confirmed by
        tokenizing the piece, so flags such as ``single_word`` are honored."""
        if self._literal_pattern is None:
            return {}
        found: set[str] = set()
        for piece in pieces:
            if piece and self._literal_pattern.search(piece) is not None:
                found |= self._literal_tokens_in(piece)
        return {"special_token_literal": sorted(found)} if found else {}

    def _literal_tokens_in(self, piece: str) -> set[str]:
        try:
            ids = self.tokenizer(piece, add_special_tokens=False)["input_ids"]
        except Exception:  # the row itself tokenized fine; this optional check stays silent
            return set()
        return {self._literal_tokens[i] for i in ids if i in self._literal_tokens}

    @staticmethod
    def message_texts(messages: Iterable[Message]) -> list[str]:
        return [piece for message in messages for piece in text_parts(message.get("content"))]

    def without_final_eos(self, text: str) -> str:
        """A plain string TRL treats as already terminated keeps its EOS text as the terminator,
        which is intended and not reported as a literal special token."""
        eos = self.eos_token
        return text[: -len(eos)] if eos and text.endswith(eos) else text

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
    "literal_token_matcher",
    "loss_positions",
    "mapping_error",
    "token_digest",
]
