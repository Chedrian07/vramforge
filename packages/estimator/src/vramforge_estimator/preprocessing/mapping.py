"""Column mapping → TRL-format example (plan.md §7.2, §7.3).

Only the mapped columns are read; every other source column is dropped and never rendered
(docs/research/example-model-dataset.md R3). The result is the exact example TRL 1.14.1 would
receive from the exported, mapped dataset, so the adapters can reproduce TRL's preprocessing.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from vramforge_estimator.schemas import ColumnMapping, EmptySystemPolicy, ErrorCode

Message = dict[str, Any]

# apply_chat_template() parameters that are not template variables. Forwarding them through
# template kwargs would change tokenization (e.g. truncation/max_length cut prompts:
# docs/research/trl-grpo.md §9, R6), so they are rejected up front.
TRUNCATION_KEYS = frozenset({"truncation", "max_length", "stride", "return_overflowing_tokens"})
RESERVED_TEMPLATE_KEYS = frozenset(
    {
        "conversation",
        "tools",
        "documents",
        "chat_template",
        "add_generation_prompt",
        "continue_final_message",
        "tokenize",
        "padding",
        "pad_to_multiple_of",
        "return_tensors",
        "return_dict",
        "return_assistant_tokens_mask",
        "tokenizer_kwargs",
        "add_special_tokens",
    }
)


class RowError(Exception):
    """One row cannot be mapped or tokenized. The message is Korean and never contains row text."""

    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def is_blank(content: Any) -> bool:
    """Empty/whitespace-only (or missing) message content."""
    return content is None or (isinstance(content, str) and not content.strip())


def content_digest(example: Mapping[str, Any]) -> str:
    """sha256 of the canonical JSON of the mapped example (not the raw row)."""
    payload = json.dumps(
        example, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def text_parts(content: Any) -> list[str]:
    """Text pieces of a message content (string or list of text blocks)."""
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        return [block["text"] for block in content if isinstance(block, dict) and "text" in block]
    return []


class RecordMapper:
    """Reads mapped columns from a raw row and builds chat messages / plain strings."""

    def __init__(self, mapping: ColumnMapping, policy: EmptySystemPolicy) -> None:
        self.mapping = mapping
        self.policy = policy

    def value(self, row: Mapping[str, Any], column: str | None, role_field: str) -> Any:
        if column is None:  # pragma: no cover - guarded by the adapters' mapping checks
            raise RowError(ErrorCode.COLUMN_MAPPING_REQUIRED, f"'{role_field}' 매핑이 없습니다.")
        if column not in row:
            raise RowError(
                ErrorCode.COLUMN_MAPPING_REQUIRED,
                f"매핑된 '{role_field}' 컬럼이 이 row에 없습니다.",
            )
        value = row[column]
        if value is None:
            raise RowError(
                ErrorCode.DATASET_FORMAT_UNSUPPORTED,
                f"매핑된 '{role_field}' 값이 비어 있습니다(null).",
            )
        return value

    def messages(
        self, row: Mapping[str, Any], column: str | None, role_field: str, role: str
    ) -> list[Message]:
        """A string becomes one `role` message; a list of message dicts is used as-is."""
        value = self.value(row, column, role_field)
        if isinstance(value, str):
            return [{"role": role, "content": value}]
        if isinstance(value, list | tuple):
            return normalize_messages(value, role_field)
        raise RowError(
            ErrorCode.DATASET_FORMAT_UNSUPPORTED,
            f"'{role_field}' 값은 문자열 또는 메시지 목록이어야 합니다 ({type(value).__name__}).",
        )

    def system_messages(self, row: Mapping[str, Any]) -> tuple[list[Message], int]:
        """The mapped system column as a message list and the number of omitted messages."""
        column = self.mapping.system
        if column is None:
            return [], 0
        if column not in row:
            raise RowError(
                ErrorCode.COLUMN_MAPPING_REQUIRED, "매핑된 'system' 컬럼이 이 row에 없습니다."
            )
        content = row[column]
        if content is not None and not isinstance(content, str):
            raise RowError(
                ErrorCode.DATASET_FORMAT_UNSUPPORTED,
                f"'system' 값은 문자열이어야 합니다 ({type(content).__name__}).",
            )
        if self.policy is EmptySystemPolicy.OMIT and is_blank(content):
            return [], 1
        return [{"role": "system", "content": content if content is not None else ""}], 0

    def apply_policy(self, messages: list[Message]) -> tuple[list[Message], int]:
        """OMIT drops system messages whose content is empty/whitespace; KEEP renders them."""
        if self.policy is not EmptySystemPolicy.OMIT:
            return messages, 0
        kept = [
            m for m in messages if not (m.get("role") == "system" and is_blank(m.get("content")))
        ]
        return kept, len(messages) - len(kept)


def normalize_messages(value: list[Any] | tuple[Any, ...], role_field: str) -> list[Message]:
    """Copy message dicts; `from`/`value` keys become `role`/`content` like TRL's
    maybe_convert_to_chatml (trl 1.14.1 data_utils.py:976-1023). Non-text content is rejected."""
    out: list[Message] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise RowError(
                ErrorCode.DATASET_FORMAT_UNSUPPORTED,
                f"'{role_field}' 목록의 항목은 메시지(dict)여야 합니다.",
            )
        message = dict(item)
        if "role" not in message and "from" in message:
            message["role"] = message.pop("from")
        if "content" not in message and "value" in message:
            message["content"] = message.pop("value")
        if not isinstance(message.get("role"), str):
            raise RowError(
                ErrorCode.DATASET_FORMAT_UNSUPPORTED,
                f"'{role_field}' 메시지에 role이 없습니다.",
            )
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if not (isinstance(block, Mapping) and isinstance(block.get("text"), str)):
                    raise RowError(
                        ErrorCode.DATASET_FORMAT_UNSUPPORTED,
                        "텍스트가 아닌 content(이미지·오디오 등)는 지원하지 않습니다.",
                    )
        elif content is not None and not isinstance(content, str):
            raise RowError(
                ErrorCode.DATASET_FORMAT_UNSUPPORTED,
                f"'{role_field}' 메시지 content는 문자열이어야 합니다.",
            )
        out.append(message)
    return out


__all__ = [
    "RESERVED_TEMPLATE_KEYS",
    "TRUNCATION_KEYS",
    "Message",
    "RecordMapper",
    "RowError",
    "content_digest",
    "is_blank",
    "normalize_messages",
    "text_parts",
]
