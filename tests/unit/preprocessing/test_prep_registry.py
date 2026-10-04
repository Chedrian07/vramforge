"""Adapter registry, mapping validation, template-kwarg guards and transformation notes."""

from __future__ import annotations

import pytest

from vramforge_estimator.errors import EstimatorError
from vramforge_estimator.preprocessing import get_adapter
from vramforge_estimator.schemas import (
    ColumnMapping,
    DatasetFormat,
    EmptySystemPolicy,
    ErrorCode,
    Objective,
)

PREF = ColumnMapping(system="system", prompt="question", chosen="chosen", rejected="rejected")


@pytest.mark.parametrize(
    ("objective", "name"),
    [
        (Objective.SFT, "trl-1.14.1-sft"),
        (Objective.DPO, "trl-1.14.1-dpo"),
        (Objective.GRPO, "trl-1.14.1-grpo"),
    ],
)
def test_adapter_names_and_versions(mimo, objective, name) -> None:
    adapter = get_adapter(objective, mimo, PREF)
    assert adapter.name == name
    assert adapter.objective is objective
    assert isinstance(adapter.version, str) and adapter.version


def test_policy_must_match_mapping(mimo) -> None:
    keep = PREF.model_copy(update={"empty_system_policy": EmptySystemPolicy.KEEP})
    with pytest.raises(EstimatorError) as err:
        get_adapter(Objective.SFT, mimo, keep)  # kwarg defaults to OMIT
    assert err.value.issue.code is ErrorCode.CONFLICTING_OPTIONS
    adapter = get_adapter(Objective.SFT, mimo, keep, empty_system_policy=EmptySystemPolicy.KEEP)
    assert adapter.empty_system_policy is EmptySystemPolicy.KEEP


@pytest.mark.parametrize("key", ["truncation", "max_length"])
def test_truncation_template_kwargs_are_rejected(mimo, key) -> None:
    with pytest.raises(EstimatorError) as err:
        get_adapter(Objective.GRPO, mimo, PREF, template_kwargs={key: 16})
    assert err.value.issue.code is ErrorCode.DATA_PRESERVATION_VIOLATION


def test_reserved_template_kwargs_are_rejected(mimo) -> None:
    with pytest.raises(EstimatorError) as err:
        get_adapter(Objective.SFT, mimo, PREF, template_kwargs={"add_generation_prompt": True})
    assert err.value.issue.code is ErrorCode.CONFLICTING_OPTIONS


@pytest.mark.parametrize(
    ("objective", "mapping"),
    [
        (Objective.SFT, ColumnMapping(prompt="q")),
        (Objective.SFT, ColumnMapping(format=DatasetFormat.PROMPT_ONLY, prompt="q")),
        (Objective.SFT, ColumnMapping(messages="m", prompt="q", completion="c")),
        (Objective.SFT, ColumnMapping(text="t", system="s")),
        (Objective.DPO, ColumnMapping(prompt="q", chosen="c")),
        (Objective.DPO, ColumnMapping(format=DatasetFormat.MESSAGES, messages="m")),
        (Objective.GRPO, ColumnMapping(messages="m")),
        (Objective.GRPO, ColumnMapping(chosen="c", rejected="r")),
    ],
)
def test_unusable_mappings_raise_column_mapping_required(mimo, objective, mapping) -> None:
    with pytest.raises(EstimatorError) as err:
        get_adapter(objective, mimo, mapping)
    assert err.value.issue.code is ErrorCode.COLUMN_MAPPING_REQUIRED


def test_messages_layout_needs_a_chat_template(plain) -> None:
    with pytest.raises(EstimatorError) as err:
        get_adapter(Objective.SFT, plain, ColumnMapping(messages="m"))
    assert err.value.issue.code is ErrorCode.TEMPLATE_REQUIRED


def test_sft_from_preference_note_states_the_data_transformation(mimo) -> None:
    note = get_adapter(Objective.SFT, mimo, PREF).transformation_note()
    assert "chosen" in note and "rejected" in note
    assert "사용하지 않습니다" in note
    assert "절단(truncation)이 아닙니다" in note
    assert "omit" in note  # system column mapped -> policy is stated


def test_grpo_note_states_prompt_only_and_no_reward_from_pairs(mimo) -> None:
    note = get_adapter(Objective.GRPO, mimo, PREF).transformation_note()
    assert "prompt만 사용" in note
    assert "보상 계산에 사용하지 않" in note


def test_dpo_note_keeps_branches_separate(mimo) -> None:
    note = get_adapter(Objective.DPO, mimo, PREF).transformation_note()
    assert "합치지 않습니다" in note
