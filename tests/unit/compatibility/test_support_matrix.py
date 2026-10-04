"""docs/support-matrix.md is generated from the profile registry and must match it (plan §20.1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from vramforge_estimator.compatibility.profiles import (
    PROFILES_DIR_ENV,
    clear_registry_cache,
    load_registry,
)
from vramforge_estimator.compatibility.support_matrix import COMMAND, main, render_support_matrix
from vramforge_estimator.schemas import Objective, Strategy

DOC = Path(__file__).resolve().parents[3] / "docs" / "support-matrix.md"


@pytest.fixture(autouse=True)
def _repo_profiles(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(PROFILES_DIR_ENV, raising=False)
    clear_registry_cache()
    yield
    clear_registry_cache()


def test_committed_matrix_matches_the_registry() -> None:
    assert DOC.read_text(encoding="utf-8") == render_support_matrix(), (
        f"docs/support-matrix.md is stale; regenerate with: {COMMAND}"
    )


def test_matrix_has_a_row_per_objective_and_adapter() -> None:
    text = render_support_matrix()
    reg = load_registry()
    for objective in Objective:
        for prof in reg.analytic.values():
            cells = [prof.support_rule(objective, s) for s in Strategy]
            row = next(
                line
                for line in text.splitlines()
                if line.startswith(f"| {objective.value} | `{prof.architecture_adapter}` |")
            )
            for rule in cells:
                assert (rule.grade.value if rule.grade else "unsupported") in row
    assert reg.environments["cuda-trl-1.14.1"].dependency_lock_digest in text


def test_cli_write_and_check(tmp_path: Path) -> None:
    out = tmp_path / "matrix.md"
    assert main(["--write", str(out)]) == 0
    assert main(["--check", str(out)]) == 0
    out.write_text("stale", encoding="utf-8")
    assert main(["--check", str(out)]) == 1
