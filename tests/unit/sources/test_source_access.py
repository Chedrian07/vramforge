"""SourceAccess never shows the user's Hugging Face token in text (plan.md §18).

The token is a field of a server-side dataclass that is passed to every resolver and inspector;
anything that formats it (reprs in tracebacks, debug logs, containers that hold it) must not
print the token, while the token itself stays available to the Hub client.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import pytest

from vramforge_estimator.sources import SourceAccess, hub

TOKEN = "hf_" + "t" * 34


def _access() -> SourceAccess:
    return SourceAccess(
        hf_token=TOKEN, local_roots={"models": Path("/srv/models")}, hf_home=Path("/srv/hf")
    )


def test_repr_and_str_never_show_the_token() -> None:
    access = _access()
    for text in (repr(access), str(access), f"{access}", f"{access!r}", format(access)):
        assert TOKEN not in text
        assert "hf_token" not in text
    # the rest of the configuration stays readable for debugging
    assert "local_roots={'models'" in repr(access)
    assert "http_timeout_s=30.0" in repr(access)
    # and the token is still there for the Hub client
    assert access.hf_token == TOKEN


def test_copies_and_containers_never_show_the_token() -> None:
    access = _access()
    copy = dataclasses.replace(access, http_timeout_s=5.0)
    assert copy.hf_token == TOKEN
    for text in (repr(copy), repr([access]), repr({"access": access}), repr((access, copy))):
        assert TOKEN not in text


def test_logging_an_access_never_writes_the_token(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("vramforge_estimator.sources.test")
    with caplog.at_level(logging.DEBUG, logger=logger.name):
        logger.debug("resolving with %s / %r", _access(), _access())
        logger.info("access=%s", {"access": _access()})
    assert caplog.records
    assert TOKEN not in caplog.text


def test_hub_client_repr_never_shows_the_token() -> None:
    client = hub.HfHubClient(_access())
    assert TOKEN not in repr(client)
    assert TOKEN not in repr(client._api)  # huggingface_hub's HfApi holds it as well
