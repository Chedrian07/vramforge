"""docs/methodology.md must describe what the code computes (plan §20.1)."""

from __future__ import annotations

import re
from pathlib import Path

from vf_fakes import (
    FakeArch,
    dpo_shape,
    grpo_shape,
    make_cfg,
    make_inventory,
    make_plan,
    sft_shape,
)

from vramforge_estimator.compatibility.profiles import load_registry
from vramforge_estimator.schemas import (
    MarginPolicy,
    Objective,
    ReferenceStrategy,
    RewardKind,
    ScopeConfig,
    Strategy,
)
from vramforge_estimator.trainers import get_trainer
from vramforge_estimator.trainers.common import DEVICE_MAP_BUDGET
from vramforge_estimator.trainers.sft import CHUNK
from vramforge_estimator.units import GiB

ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / "docs" / "methodology.md"
TRAINERS = ROOT / "packages" / "estimator" / "src" / "vramforge_estimator" / "trainers"
# Documents owned by other agents that may land after this one (docs/architecture.md §8).
OTHER_AGENT_DOCS = {"methodology-architectures.md"}


def anchors() -> set[str]:
    return set(re.findall(r'<a id="([a-z0-9-]+)"></a>', DOC.read_text(encoding="utf-8")))


def test_every_formula_reference_in_the_trainer_code_has_an_anchor() -> None:
    used: set[str] = set()
    for path in TRAINERS.glob("*.py"):
        used |= set(re.findall(r'formula="([a-z0-9-]+)"', path.read_text(encoding="utf-8")))
    assert {"sft-chunked-nll", "dpo-logits", "grpo-policy-logits", "optimizer-states"} <= used
    assert used - anchors() == set()


def test_built_schedules_only_reference_documented_formulas() -> None:
    inv = make_inventory()
    cases = [
        (make_cfg(Objective.SFT, Strategy.FULL, inventory=inv, load_dtype="float32"), sft_shape(2)),
        (make_cfg(Objective.SFT, inventory=inv, loss_path="hf_ce"), sft_shape()),
        (
            make_cfg(
                Objective.DPO,
                Strategy.FULL,
                inventory=inv,
                reference=ReferenceStrategy.STANDALONE_MODEL,
            ),
            dpo_shape(),
        ),
        (
            make_cfg(
                Objective.DPO, inventory=inv, reference=ReferenceStrategy.PRECOMPUTED_LOG_PROBS
            ),
            dpo_shape(),
        ),
        (
            make_cfg(
                Objective.GRPO,
                Strategy.FULL,
                inventory=inv,
                accumulation=4,
                beta=0.1,
                num_iterations=2,
                reward=RewardKind.LOCAL_MODEL,
            ),
            grpo_shape(),
        ),
    ]
    docs = anchors()
    scope = ScopeConfig(include_evaluation=True, include_checkpoint_save=True)
    for cfg, shape in cases:
        sched = get_trainer(cfg.objective).build_schedule(
            inv, cfg, FakeArch(), shape, make_plan(cfg, [shape]), scope
        )
        refs = {a.formula_ref for a in sched.allocations if a.formula_ref}
        assert refs, cfg.objective
        for ref in refs:
            page, _, anchor = ref.partition("#")
            assert page == "methodology.md" and anchor in docs, ref


def test_documented_constants_match_code_and_profiles() -> None:
    text = DOC.read_text(encoding="utf-8")
    policy = MarginPolicy()
    assert policy.min_bytes == 2 * GiB and "2 GiB" in text
    assert str(policy.fraction) in text
    for factor in DEVICE_MAP_BUDGET.values():
        assert f"{float(factor):g}" in text
    assert f"C = {CHUNK}" in text
    for prof in load_registry().analytic.values():
        ws = prof.workspace
        assert f"{ws.cuda_context_bytes.low / GiB:.1f} GiB" in text
        assert f"{ws.cuda_context_bytes.high / GiB:.1f} GiB" in text
        assert f"{ws.library_workspace_bytes.high / 2**20:.0f} MiB" in text
        assert f"{ws.allocator_slack_fraction.low:.0%}" in text
        assert f"{ws.allocator_slack_fraction.high:.0%}" in text


def test_relative_links_resolve() -> None:
    text = DOC.read_text(encoding="utf-8")
    for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", text):
        if target.startswith(("http://", "https://")):
            continue
        path = (DOC.parent / target).resolve()
        if Path(target).name in OTHER_AGENT_DOCS and not path.exists():
            continue
        assert path.exists(), target


def test_every_fit_reason_the_engine_returns_is_documented() -> None:
    engine = (
        ROOT / "packages" / "estimator" / "src" / "vramforge_estimator" / "memory" / "engine.py"
    )
    reasons = set(re.findall(r'reason="([a-z_]+)"', engine.read_text(encoding="utf-8")))
    assert {"load_budget_insufficient", "floor_exceeds_capacity", "unsupported"} <= reasons
    text = DOC.read_text(encoding="utf-8")
    assert {r for r in reasons if f"`{r}`" not in text} == set()


def test_load_budget_example_matches_the_engine() -> None:
    from vramforge_estimator.memory.engine import LoadBudget

    text = DOC.read_text(encoding="utf-8")
    budget = LoadBudget(s_load=7_765_103_072, quantized=True)
    assert "7,765,103,072 B" in text and "9,586,547,003 B" in text
    assert not budget.exceeded(9_586_547_003) and budget.exceeded(9_586_547_002)
    assert 'device_map={"": 0}' in text and "max_memory" in text


def test_padding_and_evidence_rules_are_documented() -> None:
    assert {"padding", "evidence", "hardware-fit"} <= anchors()
    text = DOC.read_text(encoding="utf-8")
    for needle in (
        "has_padding",
        "cat(prompt_mask, completion_mask)",
        "`evidence_level`",
        "is_embedding",
        "receives_grad",
        "peft_target_spec",
        "LOAD_BUDGET_EXCEEDED",
    ):
        assert needle in text, needle
