"""DPO adapter = TRL 1.14.1 DPO preprocessing with max_length=None (trl-sft-dpo.md §7)."""

from __future__ import annotations

from typing import Any

from vramforge_estimator.preprocessing import get_adapter
from vramforge_estimator.preprocessing.trl_dpo import extract_prompt
from vramforge_estimator.schemas import ColumnMapping, EmptySystemPolicy, ErrorCode, Objective

PREF = ColumnMapping(system="system", prompt="question", chosen="chosen", rejected="rejected")
ROW = {"system": "", "question": "C 함수 하나 작성", "chosen": "```c\nint f(void){return 0;}\n```"}


def ids(tok: Any, messages: list[dict[str, Any]], *, gen: bool = False) -> list[int]:
    return tok.apply_chat_template(messages, add_generation_prompt=gen)["input_ids"]


def u(text: str) -> dict[str, str]:
    return {"role": "user", "content": text}


def a(text: str) -> dict[str, str]:
    return {"role": "assistant", "content": text}


def test_explicit_prompt_lengths_match_trl(mimo) -> None:
    row = {**ROW, "rejected": "괜찮아요"}
    rec = get_adapter(Objective.DPO, mimo, PREF).process(row, "train:0")
    tok = mimo.tokenizer
    p = ids(tok, [u(row["question"])], gen=True)
    pc = ids(tok, [u(row["question"]), a(row["chosen"])])
    pr = ids(tok, [u(row["question"]), a(row["rejected"])])
    assert rec.ok
    assert rec.prompt_tokens == len(p)
    assert rec.chosen_total_tokens == len(p) + len(pc[len(p) :]) == len(pc)
    assert rec.rejected_total_tokens == len(pr)
    assert rec.chosen_completion_tokens == len(pc) - len(p)
    assert rec.rejected_completion_tokens == len(pr) - len(p)
    assert rec.template_preserved is True
    assert "prefix_mismatch" not in rec.extras


def test_rejected_only_outlier_is_the_longest_branch(mimo) -> None:
    row = {**ROW, "rejected": "B" * 5000}
    rec = get_adapter(Objective.DPO, mimo, PREF).process(row, "r")
    assert rec.rejected_total_tokens > rec.chosen_total_tokens
    assert rec.max_sequence_tokens() == rec.rejected_total_tokens


def test_keep_policy_adds_the_system_block_to_every_branch(mimo) -> None:
    row = {**ROW, "rejected": "no"}
    keep = PREF.model_copy(update={"empty_system_policy": EmptySystemPolicy.KEEP})
    omit_rec = get_adapter(Objective.DPO, mimo, PREF).process(row, "r")
    keep_rec = get_adapter(
        Objective.DPO, mimo, keep, empty_system_policy=EmptySystemPolicy.KEEP
    ).process(row, "r")
    delta = keep_rec.prompt_tokens - omit_rec.prompt_tokens
    assert delta > 0
    assert keep_rec.chosen_total_tokens == omit_rec.chosen_total_tokens + delta
    assert keep_rec.rejected_total_tokens == omit_rec.rejected_total_tokens + delta
    assert keep_rec.chosen_completion_tokens == omit_rec.chosen_completion_tokens


def test_implicit_prompt_message_lists_match_explicit_prompt(mimo) -> None:
    conv_c = [u("질문"), a("좋은 답")]
    conv_r = [u("질문"), a("나쁜 답")]
    implicit = get_adapter(Objective.DPO, mimo, ColumnMapping(chosen="c", rejected="r")).process(
        {"c": conv_c, "r": conv_r}, "r"
    )
    explicit = get_adapter(Objective.DPO, mimo, PREF).process(
        {"system": "", "question": "질문", "chosen": "좋은 답", "rejected": "나쁜 답"}, "r"
    )
    for field in ("prompt_tokens", "chosen_total_tokens", "rejected_total_tokens"):
        assert getattr(implicit, field) == getattr(explicit, field)


def test_extract_prompt_reproduces_trl_quirks() -> None:
    conv = [u("q"), a("same")]
    # identical lists: the loop never breaks and the last turn stays in the completions
    prompt, chosen, rejected = extract_prompt(conv, list(conv))
    assert prompt == [u("q")] and chosen == [a("same")] and rejected == [a("same")]
    prompt, chosen, rejected = extract_prompt([u("q"), a("x")], [u("q"), a("y"), u("z")])
    assert prompt == [u("q")] and chosen == [a("x")] and rejected == [a("y"), u("z")]


def test_implicit_prompt_rows_without_shared_prefix_fail(mimo) -> None:
    adapter = get_adapter(Objective.DPO, mimo, ColumnMapping(chosen="c", rejected="r"))
    rec = adapter.process({"c": [u("a"), a("x")], "r": [u("b"), a("y")]}, "r")
    assert not rec.ok and rec.error_code is ErrorCode.DATASET_FORMAT_UNSUPPORTED
    strings = adapter.process({"c": "full text a", "r": "full text b"}, "r")
    assert not strings.ok and strings.error_code is ErrorCode.DATASET_FORMAT_UNSUPPORTED


def test_plain_dpo_appends_eos_to_both_branches(plain) -> None:
    mapping = ColumnMapping(prompt="p", chosen="c", rejected="r")
    rec = get_adapter(Objective.DPO, plain, mapping).process(
        {"p": "Q:", "c": " yes", "r": " no</s>"}, "r"
    )
    tok = plain.tokenizer
    assert rec.prompt_tokens == len(tok(text="Q:")["input_ids"])
    assert rec.chosen_total_tokens == len(tok(text="Q: yes</s>")["input_ids"])
    assert rec.rejected_total_tokens == len(tok(text="Q: no</s>")["input_ids"])
    assert "duplicate_eos" not in rec.extras


def test_plain_dpo_needs_an_explicit_prompt(plain) -> None:
    adapter = get_adapter(Objective.DPO, plain, ColumnMapping(chosen="c", rejected="r"))
    rec = adapter.process({"c": "x", "r": "y"}, "r")
    assert not rec.ok and rec.error_code is ErrorCode.COLUMN_MAPPING_REQUIRED


def test_missing_rejected_column_is_a_row_error(mimo) -> None:
    rec = get_adapter(Objective.DPO, mimo, PREF).process(ROW, "train:5")
    assert not rec.ok and rec.error_code is ErrorCode.COLUMN_MAPPING_REQUIRED
