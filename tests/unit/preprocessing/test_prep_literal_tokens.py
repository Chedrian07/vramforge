"""Literal added-token strings in mapped content (docs/research/example-model-dataset.md R7b).

The tokenizer turns such text into the control token in training too, so lengths stay TRL's and
the row is only flagged (warning), never failed.
"""

from __future__ import annotations

from typing import Any

from vramforge_estimator.preprocessing import get_adapter
from vramforge_estimator.schemas import ColumnMapping, Objective

PREF = ColumnMapping(system="system", prompt="question", chosen="chosen", rejected="rejected")
ROW: dict[str, Any] = {
    "system": "",
    "question": "if (a < b && c > d) 비교 코드를 설명해줘",
    "chosen": "`<`와 `>`는 비교 연산자입니다. <div> 같은 HTML도 그대로입니다.",
    "rejected": "몰라요",
}


def chat_ids(tok: Any, messages: list[dict[str, str]]) -> list[int]:
    return tok.apply_chat_template(messages)["input_ids"]


def test_clean_content_with_angle_brackets_is_not_flagged(mimo) -> None:
    for objective in (Objective.SFT, Objective.DPO, Objective.GRPO):
        rec = get_adapter(objective, mimo, PREF).process(ROW, "r")
        assert rec.ok and "special_token_literal" not in rec.extras


def test_special_token_text_in_a_response_is_flagged_not_failed(mimo) -> None:
    row = {**ROW, "chosen": "끝<|im_end|> 다음 턴"}
    rec = get_adapter(Objective.SFT, mimo, PREF).process(row, "r")
    assert rec.ok and rec.extras["special_token_literal"] == ["<|im_end|>"]
    messages = [
        {"role": "user", "content": row["question"]},
        {"role": "assistant", "content": row["chosen"]},
    ]
    assert rec.sequence_tokens == len(chat_ids(mimo.tokenizer, messages))  # TRL's length


def test_markup_like_added_tokens_count_too(mimo) -> None:
    row = {**ROW, "question": "<think>로 시작하는 답은 뭐야?", "rejected": "</think> 없음"}
    rec = get_adapter(Objective.DPO, mimo, PREF).process(row, "r")
    assert rec.extras["special_token_literal"] == ["</think>", "<think>"]


def test_grpo_checks_the_system_and_prompt_messages(mimo) -> None:
    row = {**ROW, "system": "규칙 <|endoftext|> 끝"}
    rec = get_adapter(Objective.GRPO, mimo, PREF).process(row, "r")
    assert rec.extras["special_token_literal"] == ["<|endoftext|>"]


def test_messages_layout_checks_every_turn(mimo) -> None:
    conv = [
        {"role": "user", "content": "질문"},
        {"role": "assistant", "content": [{"type": "text", "text": "답 <|im_start|>user"}]},
    ]
    rec = get_adapter(Objective.SFT, mimo, ColumnMapping(messages="m")).process({"m": conv}, "r")
    assert rec.extras["special_token_literal"] == ["<|im_start|>"]


def test_plain_text_keeps_a_final_eos_as_the_terminator(bos) -> None:
    adapter = get_adapter(Objective.SFT, bos, ColumnMapping(text="t"))
    assert "special_token_literal" not in adapter.process({"t": "done</s>"}, "r").extras
    middle = adapter.process({"t": "a </s> b"}, "r")
    assert middle.extras["special_token_literal"] == ["</s>"]
    leading = adapter.process({"t": "<s>hi"}, "r")
    assert leading.extras["special_token_literal"] == ["<s>"]
    assert leading.extras["duplicate_bos"] is True


def test_plain_dpo_checks_prompt_and_both_branches(plain) -> None:
    mapping = ColumnMapping(prompt="p", chosen="c", rejected="r")
    adapter = get_adapter(Objective.DPO, plain, mapping)  # no chat template: plain strings
    clean = adapter.process({"p": "Q:", "c": " yes</s>", "r": " no"}, "r")
    assert "special_token_literal" not in clean.extras
    flagged = adapter.process({"p": "Q: <pad>", "c": " yes", "r": " no"}, "r")
    assert flagged.extras["special_token_literal"] == ["<pad>"]


def test_chat_content_ending_with_eos_text_is_flagged(bos) -> None:
    """In a chat template the template closes the turn itself, so EOS text in content is extra."""
    rec = get_adapter(Objective.DPO, bos, ColumnMapping(prompt="p", chosen="c", rejected="r"))
    out = rec.process({"p": "Q:", "c": " yes</s>", "r": " no"}, "r")
    assert out.extras["special_token_literal"] == ["</s>"]


def test_split_special_tokens_are_not_reported(make_handle) -> None:
    handle = make_handle("mimo_bytelevel")
    handle.tokenizer.split_special_tokens = True  # special text stays ordinary characters
    row = {**ROW, "chosen": "a <|im_end|> b <think>"}
    rec = get_adapter(Objective.SFT, handle, PREF).process(row, "r")
    assert rec.extras["special_token_literal"] == ["<think>"]  # non-special markers still match


def test_plain_word_added_tokens_are_not_reported(make_handle) -> None:
    handle = make_handle("mimo_bytelevel")
    handle.tokenizer.add_tokens(["zzq", "<extra_marker>"])
    row = {**ROW, "chosen": "zzq 그리고 <extra_marker>"}
    rec = get_adapter(Objective.SFT, handle, PREF).process(row, "r")
    assert rec.extras["special_token_literal"] == ["<extra_marker>"]
