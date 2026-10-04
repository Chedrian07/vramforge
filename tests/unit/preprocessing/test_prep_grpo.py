"""GRPO adapter = TRL 1.14.1 ``_tokenize_prompts`` (trl-grpo.md §4.1, R6)."""

from __future__ import annotations

from vramforge_estimator.preprocessing import get_adapter
from vramforge_estimator.schemas import ColumnMapping, EmptySystemPolicy, ErrorCode, Objective

PREF = ColumnMapping(system="system", prompt="question", chosen="chosen", rejected="rejected")
ROW = {
    "system": "",
    "question": "SQL injection을 막는 코드를 작성해줘",
    "chosen": "x",
    "rejected": "y",
}


def test_prompt_uses_the_generation_prompt(mimo) -> None:
    rec = get_adapter(Objective.GRPO, mimo, PREF).process(ROW, "train:0")
    expected = mimo.tokenizer.apply_chat_template(
        [{"role": "user", "content": ROW["question"]}], add_generation_prompt=True
    )["input_ids"]
    assert rec.ok and rec.prompt_tokens == len(expected)
    assert rec.completion_tokens is None and rec.sequence_tokens is None
    assert rec.max_sequence_tokens() == rec.prompt_tokens
    assert set(rec.extras) == {"system_omitted", "token_digest"}
    assert rec.extras["system_omitted"] == 1


def test_chosen_and_rejected_do_not_affect_grpo(mimo) -> None:
    adapter = get_adapter(Objective.GRPO, mimo, PREF)
    base = adapter.process(ROW, "r")
    other = adapter.process({**ROW, "chosen": "long " * 500, "rejected": "z"}, "r")
    assert other.prompt_tokens == base.prompt_tokens
    assert other.content_digest == base.content_digest


def test_enable_thinking_false_adds_two_prompt_tokens(mimo) -> None:
    base = get_adapter(Objective.GRPO, mimo, PREF).process(ROW, "r")
    nothink = get_adapter(
        Objective.GRPO, mimo, PREF, template_kwargs={"enable_thinking": False}
    ).process(ROW, "r")
    assert nothink.prompt_tokens == base.prompt_tokens + 2


def test_keep_policy_counts_the_empty_system_block(mimo) -> None:
    keep = PREF.model_copy(update={"empty_system_policy": EmptySystemPolicy.KEEP})
    rec = get_adapter(Objective.GRPO, mimo, keep, empty_system_policy=EmptySystemPolicy.KEEP)
    base = get_adapter(Objective.GRPO, mimo, PREF).process(ROW, "r")
    assert rec.process(ROW, "r").prompt_tokens > base.prompt_tokens


def test_message_list_prompts_are_used_as_is(mimo) -> None:
    conv = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    rec = get_adapter(Objective.GRPO, mimo, ColumnMapping(prompt="p")).process({"p": conv}, "r")
    expected = mimo.tokenizer.apply_chat_template(conv, add_generation_prompt=True)["input_ids"]
    assert rec.prompt_tokens == len(expected)


def test_plain_prompts_use_the_tokenizer_defaults(plain) -> None:
    mapping = ColumnMapping(prompt="p")
    rec = get_adapter(Objective.GRPO, plain, mapping).process({"p": "Q: 2+2?"}, "r")
    assert rec.prompt_tokens == len(b"Q: 2+2?")  # no EOS for prompts
    assert rec.template_preserved is None


def test_null_prompt_is_a_row_error(mimo) -> None:
    rec = get_adapter(Objective.GRPO, mimo, PREF).process({**ROW, "question": None}, "r")
    assert not rec.ok and rec.error_code is ErrorCode.DATASET_FORMAT_UNSUPPORTED
