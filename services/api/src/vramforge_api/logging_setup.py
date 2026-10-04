"""Logging with secret redaction (plan.md §18: no tokens in logs).

The redaction runs on the fully formatted record (message, arguments and traceback), so a token
that reaches a log line through an exception message is still masked.
"""

from __future__ import annotations

import logging
import logging.config
import re
from typing import Any

_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # Hugging Face user/org tokens.
    (re.compile(r"\bhf_[A-Za-z0-9]{8,}"), "hf_***"),
    # Authorization headers and bearer tokens.
    (re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._~+/=-]{4,}"), r"\1 ***"),
    (re.compile(r"(?i)(authorization['\"]?\s*[:=]\s*['\"]?)[^'\"\s,}]+"), r"\1***"),
    # Session cookies issued by this service.
    (re.compile(r"\b(vf_owner|vf_access)=([^;\s,'\"]+)"), r"\1=***"),
    # Credentials and signatures in URLs.
    (re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)([^/\s:@]+):([^/\s@]+)@"), r"\1\2:***@"),
    (
        re.compile(
            r"(?i)([?&](?:token|access_token|api_key|apikey|signature|sig|x-amz-signature|"
            r"x-amz-credential|x-amz-security-token|expires|key-pair-id|policy)=)[^&\s'\"]+"
        ),
        r"\1***",
    ),
)


def redact(text: str) -> str:
    """Mask secrets (HF tokens, bearer tokens, cookies, URL credentials) in a string."""
    for pattern, repl in _PATTERNS:
        text = pattern.sub(repl, text)
    return text


class RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))


class RedactingFilter(logging.Filter):
    """Redacts message/args before other handlers see them (defense in depth)."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # malformed args: leave the record to the formatter
            return True
        cleaned = redact(message)
        if cleaned != message:
            record.msg = cleaned
            record.args = None
        return True


def logging_config(level: str = "INFO") -> dict[str, Any]:
    """dictConfig used by the API (also passed to uvicorn) and the worker."""
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    handler = {
        "class": "logging.StreamHandler",
        "formatter": "redacting",
        "filters": ["redact"],
        "stream": "ext://sys.stderr",
    }
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {"redact": {"()": RedactingFilter}},
        "formatters": {
            "redacting": {"()": RedactingFormatter, "fmt": fmt},
            "access": {"()": RedactingFormatter, "fmt": "%(asctime)s %(levelname)s %(message)s"},
        },
        "handlers": {
            "default": handler,
            "access": {**handler, "formatter": "access", "stream": "ext://sys.stdout"},
        },
        "loggers": {
            "uvicorn": {"handlers": ["default"], "level": level, "propagate": False},
            "uvicorn.error": {"level": level},
            "uvicorn.access": {"handlers": ["access"], "level": level, "propagate": False},
            "rq": {"handlers": ["default"], "level": level, "propagate": False},
        },
        "root": {"handlers": ["default"], "level": level},
    }


def configure_logging(level: str = "INFO") -> None:
    logging.config.dictConfig(logging_config(level.upper()))


__all__ = ["RedactingFilter", "RedactingFormatter", "configure_logging", "logging_config", "redact"]
