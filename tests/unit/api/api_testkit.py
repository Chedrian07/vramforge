"""Helpers shared by the API tests (uniquely named: test dirs are not packages)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
EXAMPLE_REQUEST = REPO / "tests" / "fixtures" / "requests" / "plan_example_grpo.json"
CSRF = {"X-VramForge-Request": "1"}


def example_request(**overrides: Any) -> dict[str, Any]:
    """The plan §15.2 example request with dotted-path overrides."""
    data = json.loads(EXAMPLE_REQUEST.read_text(encoding="utf-8"))
    for dotted, value in overrides.items():
        node = data
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node[key]
        node[leaf] = value
    return data
