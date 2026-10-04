"""Column kinds and mapping candidates: format detection, ranking per objective, ambiguity."""

from __future__ import annotations

import pyarrow as pa
import pytest

from vramforge_estimator.inspection.dataset_mapping import analyze_mapping
from vramforge_estimator.inspection.dataset_schema import (
    Preview,
    arrow_dtype,
    preview_columns,
    value_kind,
)
from vramforge_estimator.schemas import (
    ColumnMapping,
    DatasetColumn,
    DatasetFormat,
    EmptySystemPolicy,
    ErrorCode,
    Objective,
)

EXAMPLE_COLUMNS = ["lang", "vulnerability", "system", "question", "chosen", "rejected"]
EXAMPLE_MAPPING = ColumnMapping(
    format=DatasetFormat.PREFERENCE,
    system="system",
    prompt="question",
    chosen="chosen",
    rejected="rejected",
)
OBJECTIVES = [None, Objective.SFT, Objective.DPO, Objective.GRPO]


def cols(*specs: str) -> list[DatasetColumn]:
    """`name` (string kind) or `name:kind`."""
    out = []
    for spec in specs:
        name, _, kind = spec.partition(":")
        out.append(DatasetColumn(name=name, dtype="string", kind=kind or "string"))
    return out


def codes(analysis: object) -> list[ErrorCode]:
    return [issue.code for issue in analysis.issues]  # type: ignore[attr-defined]


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_example_dataset_suggestion_is_exact(objective: Objective | None) -> None:
    analysis = analyze_mapping(cols(*EXAMPLE_COLUMNS), objective)
    assert analysis.suggested == EXAMPLE_MAPPING
    assert analysis.suggested.empty_system_policy == EmptySystemPolicy.OMIT
    assert analysis.candidates == [EXAMPLE_MAPPING]
    assert not analysis.ambiguous
    assert analysis.detected_format == DatasetFormat.PREFERENCE
    assert analysis.issues == []


def test_metadata_columns_are_never_mapped() -> None:
    analysis = analyze_mapping(cols(*EXAMPLE_COLUMNS, "input_length:number", "source"), None)
    for mapping in analysis.candidates:
        used = {mapping.system, mapping.prompt, mapping.chosen, mapping.rejected}
        assert not used & {"lang", "vulnerability", "source", "input_length"}


def test_conversational_and_prompt_completion_rank_by_objective() -> None:
    columns = cols("prompt", "completion", "messages:messages")
    sft = analyze_mapping(columns, Objective.SFT)
    assert sft.candidates[0].format == DatasetFormat.MESSAGES
    assert sft.ambiguous and sft.suggested is None
    assert codes(sft) == [ErrorCode.COLUMN_MAPPING_REQUIRED]

    grpo = analyze_mapping(columns, Objective.GRPO)
    assert not grpo.ambiguous
    assert grpo.suggested == ColumnMapping(
        format=DatasetFormat.PROMPT_COMPLETION, prompt="prompt", completion="completion"
    )

    dpo = analyze_mapping(columns, Objective.DPO)
    assert dpo.suggested is None and not dpo.ambiguous
    assert dpo.issues[0].details["reason"] == "no_candidate"


def test_ambiguity_depends_on_the_columns_the_objective_consumes() -> None:
    columns = cols("question", "answer", "chosen", "rejected")
    dpo = analyze_mapping(columns, Objective.DPO)
    assert not dpo.ambiguous and dpo.suggested is not None
    assert dpo.suggested.format == DatasetFormat.PREFERENCE
    grpo = analyze_mapping(columns, Objective.GRPO)
    assert not grpo.ambiguous  # both readings use prompt=question
    sft = analyze_mapping(columns, Objective.SFT)
    assert sft.ambiguous  # chosen vs answer changes the trained completion
    assert sft.issues[0].details["conflicting_roles"]


@pytest.mark.parametrize("objective", [Objective.SFT, Objective.DPO, Objective.GRPO])
def test_two_prompt_columns_are_ambiguous(objective: Objective) -> None:
    analysis = analyze_mapping(cols("prompt", "question", "chosen", "rejected"), objective)
    assert analysis.ambiguous
    assert analysis.suggested is None
    assert "prompt" in analysis.issues[0].details["conflicting_roles"]
    prompts = {m.prompt for m in analysis.candidates if m.format == DatasetFormat.PREFERENCE}
    assert prompts == {"prompt", "question"}


def test_two_system_columns_are_ambiguous() -> None:
    analysis = analyze_mapping(
        cols("system", "system_prompt", "prompt", "completion"), Objective.SFT
    )
    assert analysis.ambiguous
    assert analysis.issues[0].details["conflicting_roles"] == ["system"]


@pytest.mark.parametrize(
    ("columns", "objective", "expected"),
    [
        (("text",), Objective.SFT, ColumnMapping(format=DatasetFormat.TEXT, text="text")),
        (
            ("prompt",),
            Objective.GRPO,
            ColumnMapping(format=DatasetFormat.PROMPT_ONLY, prompt="prompt"),
        ),
        (
            ("chosen:messages", "rejected:messages"),
            Objective.DPO,
            ColumnMapping(format=DatasetFormat.PREFERENCE, chosen="chosen", rejected="rejected"),
        ),
        (
            ("conversations:messages", "id:number"),
            Objective.SFT,
            ColumnMapping(format=DatasetFormat.MESSAGES, messages="conversations"),
        ),
        (
            ("Instruction", "Output", "category"),
            Objective.SFT,
            ColumnMapping(
                format=DatasetFormat.PROMPT_COMPLETION, prompt="Instruction", completion="Output"
            ),
        ),
    ],
)
def test_single_reading_is_suggested(
    columns: tuple[str, ...], objective: Objective, expected: ColumnMapping
) -> None:
    analysis = analyze_mapping(cols(*columns), objective)
    assert analysis.suggested == expected
    assert not analysis.ambiguous


@pytest.mark.parametrize(
    ("columns", "objective"),
    [(("text",), Objective.DPO), (("text",), Objective.GRPO), (("prompt",), Objective.SFT)],
)
def test_no_usable_candidate_asks_for_a_mapping(
    columns: tuple[str, ...], objective: Objective
) -> None:
    analysis = analyze_mapping(cols(*columns), objective)
    assert analysis.suggested is None
    assert codes(analysis) == [ErrorCode.COLUMN_MAPPING_REQUIRED]


def test_column_kind_must_fit_the_role() -> None:
    analysis = analyze_mapping(cols("prompt:number", "completion"), Objective.SFT)
    assert analysis.suggested is None  # a numeric "prompt" is not a prompt


def test_requested_hints_resolve_ambiguity() -> None:
    columns = cols("prompt", "question", "chosen", "rejected")
    hint = ColumnMapping(prompt="question")
    analysis = analyze_mapping(columns, Objective.DPO, hint)
    assert not analysis.ambiguous
    assert analysis.suggested == ColumnMapping(
        format=DatasetFormat.PREFERENCE, prompt="question", chosen="chosen", rejected="rejected"
    )


def test_plan_request_mapping_resolves_to_preference() -> None:
    requested = ColumnMapping(
        system="system", prompt="question", chosen="chosen", rejected="rejected"
    )
    analysis = analyze_mapping(cols(*EXAMPLE_COLUMNS), Objective.GRPO, requested)
    assert analysis.suggested == EXAMPLE_MAPPING


def test_hint_may_name_a_non_synonym_column() -> None:
    requested = ColumnMapping(prompt="task", empty_system_policy=EmptySystemPolicy.KEEP)
    analysis = analyze_mapping(cols("task", "chosen", "rejected", "lang"), Objective.DPO, requested)
    assert analysis.suggested == ColumnMapping(
        format=DatasetFormat.PREFERENCE,
        prompt="task",
        chosen="chosen",
        rejected="rejected",
        empty_system_policy=EmptySystemPolicy.KEEP,
    )


def test_explicit_mapping_is_validated_not_guessed() -> None:
    columns = cols("task", "good", "bad", "lang")
    explicit = ColumnMapping(
        format=DatasetFormat.PREFERENCE, prompt="task", chosen="good", rejected="bad"
    )
    accepted = analyze_mapping(columns, Objective.DPO, explicit)
    assert accepted.suggested == explicit and accepted.candidates[0] == explicit

    missing = ColumnMapping(
        format=DatasetFormat.PREFERENCE, prompt="task", chosen="nope", rejected="bad"
    )
    rejected = analyze_mapping(columns, Objective.DPO, missing)
    assert rejected.suggested is None
    assert rejected.issues[0].details["reason"] == "column_not_found"

    incomplete = ColumnMapping(format=DatasetFormat.PROMPT_COMPLETION, prompt="task")
    assert analyze_mapping(columns, Objective.SFT, incomplete).issues[0].details["reason"] == (
        "roles_do_not_match_format"
    )

    wrong_objective = ColumnMapping(
        format=DatasetFormat.PROMPT_COMPLETION, prompt="task", completion="good"
    )
    result = analyze_mapping(columns, Objective.DPO, wrong_objective)
    assert result.suggested is None
    assert result.issues[-1].details["reason"] == "format_not_usable"


def test_value_kinds() -> None:
    assert value_kind(["a", None, ""]) == "string"
    assert value_kind([[{"role": "user", "content": "x"}], None]) == "messages"
    assert value_kind([[{"from": "human", "value": "x"}]]) == "messages"
    assert value_kind([[], [{"role": "user"}]]) == "other"  # empty conversation is not messages
    assert value_kind([1, 2.5]) == "number"
    assert value_kind([True]) == "other"
    assert value_kind([None], pa.string()) == "string"  # all-null preview falls back to the type
    assert value_kind([None], pa.list_(pa.struct([("role", pa.string())]))) == "messages"


def test_preview_columns_keep_schema_order_and_dtypes() -> None:
    from datasets import Features, Json, List, Value

    features = Features(
        {"messages": List(Json()), "score": Value("float64"), "system": Value("string")}
    )
    preview = Preview(
        rows=[
            {"messages": [{"role": "user", "content": "x"}], "score": 1.0, "system": None},
            {"messages": [{"role": "assistant", "content": "y"}], "score": None, "system": None},
        ],
        features=features,
    )
    columns = preview_columns(preview)
    assert [(c.name, c.dtype, c.kind) for c in columns] == [
        ("messages", "list<json>", "messages"),
        ("score", "double", "number"),
        ("system", "string", "string"),
    ]
    assert arrow_dtype(pa.list_(pa.struct([("role", pa.string())]))) == "list<struct<role: string>>"
