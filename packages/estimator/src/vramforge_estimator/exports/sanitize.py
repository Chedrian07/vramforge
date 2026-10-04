"""Export redaction (plan.md §12.4, §18): no tokens, absolute paths or private URLs leave the
service in an export, whatever a module put into a string field.

Applied to every string value of an export (defense in depth; results are expected to be clean
already). Public URLs keep scheme, host and path but lose credentials, query and fragment.
"""

from __future__ import annotations

import ipaddress
import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit

REDACTED_PATH = "<redacted-path>"
REDACTED_URL = "<redacted-url>"

_HF_TOKEN = re.compile(r"\bhf_[A-Za-z0-9]{8,}")
_GH_TOKEN = re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}")
_BEARER = re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._~+/=-]{4,}")
_URL = re.compile(r"\b[a-zA-Z][a-zA-Z0-9+.-]*://[^\s'\"<>`]+")
# An absolute POSIX path with at least two segments that does not continue a word, a URL or a
# relative path ("text/event-stream", "P50/P90" and "docs/x.md" are left alone).
_POSIX_PATH = re.compile(r"(?<![\w.:/~\\-])/(?:[^\s/'\"<>|`]+/)+[^\s/'\"<>|`]*")
_WINDOWS_PATH = re.compile(r"\b[A-Za-z]:\\[^\s'\"<>|`]+")
_HOME_PATH = re.compile(r"(?<![\w/])~/[^\s'\"<>|`]+")

_PRIVATE_SUFFIXES = (".local", ".internal", ".lan", ".home", ".corp", ".localdomain")
_PRIVATE_NAMES = {"localhost", "metadata", "metadata.google.internal"}


def _private_host(host: str) -> bool:
    name = host.strip("[]").lower()
    if not name:
        return True
    if name in _PRIVATE_NAMES or name.endswith(_PRIVATE_SUFFIXES) or "." not in name:
        return True
    try:
        address = ipaddress.ip_address(name)
    except ValueError:
        return False
    return not address.is_global


def _clean_url(match: re.Match[str]) -> str:
    raw = match.group(0).rstrip(".,;:)]}")
    tail = match.group(0)[len(raw) :]
    try:
        parts = urlsplit(raw)
        host = parts.hostname or ""
    except ValueError:
        return REDACTED_URL + tail
    if parts.scheme.lower() not in ("http", "https") or _private_host(host):
        return REDACTED_URL + tail
    netloc = host if parts.port is None else f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, netloc, parts.path, "", "")) + tail


def redact_text(text: str) -> str:
    text = _HF_TOKEN.sub("hf_***", text)
    text = _GH_TOKEN.sub("gh_***", text)
    text = _BEARER.sub(r"\1 ***", text)
    text = _URL.sub(_clean_url, text)
    text = _WINDOWS_PATH.sub(REDACTED_PATH, text)
    text = _HOME_PATH.sub(REDACTED_PATH, text)
    return _POSIX_PATH.sub(REDACTED_PATH, text)


def sanitize(value: Any) -> Any:
    """Recursively redact every string (dict keys included) of a JSON-like structure."""
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {sanitize(k) if isinstance(k, str) else k: sanitize(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [sanitize(v) for v in value]
    return value


__all__ = ["REDACTED_PATH", "REDACTED_URL", "redact_text", "sanitize"]
