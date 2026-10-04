"""Secrets never reach log output (plan §18)."""

import io
import logging

from vramforge_api.logging_setup import RedactingFilter, RedactingFormatter, redact

# Token-shaped strings are assembled at runtime so no secret-like literal is committed.
FAKE_HF_TOKEN = "hf_" + "AbCd" * 6


def test_redact_masks_tokens_cookies_and_url_credentials() -> None:
    text = (
        f"token {FAKE_HF_TOKEN} used with Authorization: Bearer abc.def-ghi "
        "cookie vf_owner=SECRETVALUE123; url https://user:pass@example.com/x?sig=XYZ&a=1"
    )
    out = redact(text)
    assert FAKE_HF_TOKEN not in out
    assert "abc.def-ghi" not in out
    assert "SECRETVALUE123" not in out
    assert "pass@" not in out and "sig=XYZ" not in out
    assert "a=1" in out  # ordinary query parameters stay readable


def test_formatter_redacts_tracebacks() -> None:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(RedactingFormatter("%(message)s"))
    handler.addFilter(RedactingFilter())
    logger = logging.getLogger("vf-test-redaction")
    logger.addHandler(handler)
    logger.propagate = False
    try:
        try:
            raise RuntimeError(f"failed with {FAKE_HF_TOKEN}")
        except RuntimeError:
            logger.exception("call with %s", "Bearer sekrit-value")
    finally:
        logger.removeHandler(handler)
    out = stream.getvalue()
    assert FAKE_HF_TOKEN not in out
    assert "sekrit-value" not in out
    assert "RuntimeError" in out
