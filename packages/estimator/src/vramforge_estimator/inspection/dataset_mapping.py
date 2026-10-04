"""Column-mapping candidates ranked for an objective (plan.md §4.1, §7.2).

Only columns whose (case-insensitive) name is a known synonym of a role are ever mapped, so
metadata columns (e.g. `lang`, `vulnerability`) never reach a prompt. Every combination of role
columns that forms a TRL dataset format (docs/research/trl-sft-dpo.md §2: preference,
prompt-completion, prompt-only, conversational `messages`, language-modeling `text`) is a
candidate. Candidates are ranked for the objective; the mapping is ambiguous when two usable
candidates differ in the columns the objective actually consumes (DPO: system/prompt/chosen/
rejected, SFT: the rendered conversation, GRPO: system/prompt), because then the choice changes the
analysis. Ambiguity never resolves to a guess: `suggested` stays None and the issue asks the user.

A request mapping with an explicit `format` is taken as is (validated); with `format=auto` its
roles are fixed hints and the remaining roles are detected.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from vramforge_estimator.errors import make_issue
from vramforge_estimator.schemas import (
    ColumnMapping,
    DatasetColumn,
    DatasetFormat,
    EmptySystemPolicy,
    ErrorCode,
    Issue,
    Objective,
    Severity,
    Stage,
)

ROLES = ("system", "prompt", "chosen", "rejected", "completion", "messages", "text")
SYNONYMS: dict[str, tuple[str, ...]] = {
    "prompt": ("prompt", "question", "instruction", "input", "query"),
    "chosen": ("chosen", "preferred", "accepted", "winner"),
    "rejected": ("rejected", "dispreferred", "loser"),
    "system": ("system", "system_prompt"),
    "completion": ("completion", "response", "output", "answer"),
    "messages": ("messages", "conversations", "conversation", "chat"),
    "text": ("text", "content"),
}
ROLE_KINDS: dict[str, frozenset[str]] = {
    "system": frozenset({"string"}),
    "prompt": frozenset({"string", "messages"}),
    "chosen": frozenset({"string", "messages"}),
    "rejected": frozenset({"string", "messages"}),
    "completion": frozenset({"string", "messages"}),
    "messages": frozenset({"messages"}),
    "text": frozenset({"string"}),
}
FORMAT_ROLES: dict[DatasetFormat, tuple[str, ...]] = {
    DatasetFormat.PREFERENCE: ("system", "prompt", "chosen", "rejected"),
    DatasetFormat.PROMPT_COMPLETION: ("system", "prompt", "completion"),
    DatasetFormat.PROMPT_ONLY: ("system", "prompt"),
    DatasetFormat.MESSAGES: ("messages",),
    DatasetFormat.TEXT: ("text",),
}
REQUIRED_ROLES: dict[DatasetFormat, tuple[str, ...]] = {
    DatasetFormat.PREFERENCE: ("chosen", "rejected"),
    DatasetFormat.PROMPT_COMPLETION: ("prompt", "completion"),
    DatasetFormat.PROMPT_ONLY: ("prompt",),
    DatasetFormat.MESSAGES: ("messages",),
    DatasetFormat.TEXT: ("text",),
}
# Format keys: DatasetFormat values plus "preference_implicit" (no prompt column; TRL extracts it).
_FORMAT_RANK: dict[Objective | None, dict[str, int]] = {
    None: {
        "preference": 0,
        "prompt_completion": 1,
        "messages": 2,
        "prompt_only": 3,
        "text": 4,
        "preference_implicit": 5,
    },
    Objective.SFT: {"messages": 0, "prompt_completion": 1, "preference": 2, "text": 3},
    Objective.DPO: {"preference": 0, "preference_implicit": 1},
    Objective.GRPO: {"preference": 0, "prompt_completion": 1, "prompt_only": 2},
}
_OBJECTIVE_NEEDS = {
    Objective.SFT: (
        "SFT에는 prompt·completion(또는 chosen), messages 또는 text 역할의 컬럼이 필요합니다."
    ),
    Objective.DPO: "DPO에는 prompt·chosen·rejected 역할의 컬럼이 필요합니다.",
    Objective.GRPO: "GRPO에는 prompt 역할의 컬럼이 필요합니다.",
}


@dataclass
class MappingAnalysis:
    candidates: list[ColumnMapping] = field(default_factory=list)
    suggested: ColumnMapping | None = None
    ambiguous: bool = False
    detected_format: DatasetFormat | None = None
    issues: list[Issue] = field(default_factory=list)


def analyze_mapping(
    columns: list[DatasetColumn],
    objective: Objective | None,
    requested: ColumnMapping | None = None,
) -> MappingAnalysis:
    kinds = {column.name: column.kind for column in columns}
    order = {column.name: i for i, column in enumerate(columns)}
    policy = requested.empty_system_policy if requested is not None else EmptySystemPolicy.OMIT
    detected = _detected_format(columns, order, policy)
    if requested is not None and requested.format != DatasetFormat.AUTO:
        return _explicit(requested, kinds, order, objective, detected, columns)
    hints = _given_roles(requested)
    missing = sorted({column for column in hints.values() if column not in kinds})
    if missing:
        return MappingAnalysis(
            candidates=_ranked(_candidates(columns, {}, policy), objective, order),
            detected_format=detected,
            issues=[_missing_columns_issue(missing)],
        )
    issues = _kind_issues(hints, kinds)
    ranked = _ranked(_candidates(columns, hints, policy), objective, order)
    usable = [mapping for mapping in ranked if _usable(mapping, objective)]
    analysis = MappingAnalysis(candidates=ranked, detected_format=detected, issues=issues)
    if not usable:
        analysis.issues.append(_no_candidate_issue(objective, columns))
        return analysis
    analysis.ambiguous = _is_ambiguous(usable, objective)
    if analysis.ambiguous:
        analysis.issues.append(_ambiguous_issue(usable, objective))
    elif not issues:
        analysis.suggested = usable[0]
        analysis.detected_format = usable[0].format  # the format the analysis will use
    return analysis


# ---------------------------------------------------------------- candidates


def _given_roles(mapping: ColumnMapping | None) -> dict[str, str]:
    if mapping is None:
        return {}
    return {role: value for role in ROLES if (value := getattr(mapping, role))}


def _role_matches(columns: list[DatasetColumn], forced: dict[str, str]) -> dict[str, list[str]]:
    taken = set(forced.values())
    matches: dict[str, list[str]] = {}
    for role in ROLES:
        if role in forced:
            matches[role] = [forced[role]]
            continue
        synonyms = SYNONYMS[role]
        found = [
            column
            for column in columns
            if column.name not in taken
            and column.name.strip().lower() in synonyms
            and column.kind in ROLE_KINDS[role]
        ]
        found.sort(key=lambda column: synonyms.index(column.name.strip().lower()))
        matches[role] = [column.name for column in found]
    return matches


def _candidates(
    columns: list[DatasetColumn], forced: dict[str, str], policy: EmptySystemPolicy
) -> list[ColumnMapping]:
    matches = _role_matches(columns, forced)
    systems: list[str | None] = list(matches["system"]) or [None]
    prompts = matches["prompt"]
    out: list[ColumnMapping] = []

    def add(fmt: DatasetFormat, **roles: str | None) -> None:
        used = [value for value in roles.values() if value]
        if len(used) != len(set(used)):
            return  # one column cannot play two roles
        mapping = ColumnMapping(format=fmt, empty_system_policy=policy, **roles)
        if mapping not in out:
            out.append(mapping)

    for chosen, rejected in itertools.product(matches["chosen"], matches["rejected"]):
        if prompts:
            for prompt, system in itertools.product(prompts, systems):
                add(
                    DatasetFormat.PREFERENCE,
                    system=system,
                    prompt=prompt,
                    chosen=chosen,
                    rejected=rejected,
                )
        else:
            add(DatasetFormat.PREFERENCE, chosen=chosen, rejected=rejected)
    for prompt, completion, system in itertools.product(prompts, matches["completion"], systems):
        add(DatasetFormat.PROMPT_COMPLETION, system=system, prompt=prompt, completion=completion)
    for messages in matches["messages"]:
        add(DatasetFormat.MESSAGES, messages=messages)
    for text in matches["text"]:
        add(DatasetFormat.TEXT, text=text)
    if not (matches["chosen"] or matches["rejected"] or matches["completion"]):
        for prompt, system in itertools.product(prompts, systems):
            add(DatasetFormat.PROMPT_ONLY, system=system, prompt=prompt)
    return out


def _format_key(mapping: ColumnMapping) -> str:
    if mapping.format == DatasetFormat.PREFERENCE and mapping.prompt is None:
        return "preference_implicit"
    return mapping.format.value


def _usable(mapping: ColumnMapping, objective: Objective | None) -> bool:
    return _format_key(mapping) in _FORMAT_RANK[objective]


def _ranked(
    candidates: list[ColumnMapping], objective: Objective | None, order: dict[str, int]
) -> list[ColumnMapping]:
    ranks = _FORMAT_RANK[objective]

    def key(mapping: ColumnMapping) -> tuple[int, int, int, tuple[int, ...]]:
        synonym_rank = 0
        positions = []
        for role in ROLES:
            column = getattr(mapping, role)
            if column is None:
                continue
            name = column.strip().lower()
            synonym_rank += SYNONYMS[role].index(name) if name in SYNONYMS[role] else 0
            positions.append(order.get(column, len(order)))
        usable = _format_key(mapping) in ranks
        return (
            0 if usable else 1,
            ranks.get(_format_key(mapping), 99),
            synonym_rank,
            tuple(positions),
        )

    return sorted(candidates, key=key)


def _projection(mapping: ColumnMapping, objective: Objective | None) -> tuple[str | None, ...]:
    """The columns the objective consumes: equal projections give equal analyses."""
    key = _format_key(mapping)
    if objective == Objective.DPO:
        return (mapping.system, mapping.prompt, mapping.chosen, mapping.rejected)
    if objective == Objective.GRPO:
        return (mapping.system, mapping.prompt)
    if objective == Objective.SFT:
        if key == "preference":
            return ("conversation", mapping.system, mapping.prompt, mapping.chosen)
        if key == "prompt_completion":
            return ("conversation", mapping.system, mapping.prompt, mapping.completion)
        return (key, mapping.messages, mapping.text)
    return (key, *(getattr(mapping, role) for role in ROLES))


def _is_ambiguous(usable: list[ColumnMapping], objective: Objective | None) -> bool:
    if objective is None:
        top = _format_key(usable[0])
        usable = [mapping for mapping in usable if _format_key(mapping) == top]
    return len({_projection(mapping, objective) for mapping in usable}) > 1


def _detected_format(
    columns: list[DatasetColumn], order: dict[str, int], policy: EmptySystemPolicy
) -> DatasetFormat | None:
    ranked = _ranked(_candidates(columns, {}, policy), None, order)
    return ranked[0].format if ranked else None


# ---------------------------------------------------------------- explicit request


def _explicit(
    requested: ColumnMapping,
    kinds: dict[str, str],
    order: dict[str, int],
    objective: Objective | None,
    detected: DatasetFormat | None,
    columns: list[DatasetColumn],
) -> MappingAnalysis:
    fmt = requested.format
    given = _given_roles(requested)
    auto = _ranked(_candidates(columns, {}, requested.empty_system_policy), objective, order)
    analysis = MappingAnalysis(candidates=auto, detected_format=detected)
    missing = sorted({column for column in given.values() if column not in kinds})
    if missing:
        analysis.issues.append(_missing_columns_issue(missing))
        return analysis
    extra = sorted(role for role in given if role not in FORMAT_ROLES[fmt])
    absent = [role for role in REQUIRED_ROLES[fmt] if role not in given]
    if extra or absent:
        analysis.issues.append(
            _mapping_issue(
                f"{fmt.value} 형식의 매핑에 필요한 역할이 빠졌거나 쓰이지 않는 역할이 있습니다.",
                reason="roles_do_not_match_format",
                missing_roles=absent,
                unused_roles=extra,
            )
        )
        return analysis
    mapping = ColumnMapping(format=fmt, empty_system_policy=requested.empty_system_policy, **given)
    analysis.issues.extend(_kind_issues(given, kinds))
    if not _usable(mapping, objective):
        analysis.issues.append(
            _mapping_issue(
                f"지정한 {fmt.value} 매핑은 선택한 학습 방식에 쓸 수 없습니다. "
                + _OBJECTIVE_NEEDS.get(objective, ""),  # type: ignore[arg-type]
                reason="format_not_usable",
            )
        )
    analysis.candidates = [mapping, *(m for m in auto if m != mapping)]
    if not analysis.issues:
        analysis.suggested = mapping
        analysis.detected_format = fmt
    return analysis


# ---------------------------------------------------------------- issues


def _kind_issues(given: dict[str, str], kinds: dict[str, str]) -> list[Issue]:
    wrong = {
        role: column
        for role, column in given.items()
        if kinds.get(column) not in ROLE_KINDS[role] and kinds.get(column) != "other"
    }
    if not wrong:
        return []
    return [
        _mapping_issue(
            "컬럼 값의 형식이 지정한 역할에 맞지 않습니다: "
            + ", ".join(f"{column}→{role}" for role, column in wrong.items()),
            reason="kind_mismatch",
            roles=wrong,
        )
    ]


def _mapping_issue(message: str, **details: object) -> Issue:
    return make_issue(
        ErrorCode.COLUMN_MAPPING_REQUIRED,
        message,
        severity=Severity.ERROR,
        stage=Stage.INSPECTING,
        component="dataset.mapping",
        **details,
    )


def _missing_columns_issue(missing: list[str]) -> Issue:
    return _mapping_issue(
        "지정한 컬럼이 데이터셋에 없습니다: " + ", ".join(missing),
        reason="column_not_found",
        missing_columns=missing,
    )


def _no_candidate_issue(objective: Objective | None, columns: list[DatasetColumn]) -> Issue:
    needs = _OBJECTIVE_NEEDS.get(objective, "") if objective is not None else ""  # type: ignore[call-overload]
    return _mapping_issue(
        "컬럼 역할을 자동으로 정할 수 없습니다. " + needs + " 컬럼 매핑을 지정해 주세요.",
        reason="no_candidate",
        columns=[column.name for column in columns],
    )


def _ambiguous_issue(usable: list[ColumnMapping], objective: Objective | None) -> Issue:
    roles = sorted(
        {role for role in ROLES if len({getattr(mapping, role) for mapping in usable}) > 1}
    )
    return _mapping_issue(
        "여러 컬럼 조합이 가능하고 선택에 따라 분석 결과가 달라집니다. "
        "사용할 매핑을 선택해 주세요.",
        reason="ambiguous",
        conflicting_roles=roles,
        candidate_count=len(usable),
        objective=objective.value if objective is not None else None,
    )
