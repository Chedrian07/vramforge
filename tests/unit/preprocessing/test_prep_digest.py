"""Token digests: a hash of the ids the trainer feeds to the model, per row (plan §7.6).

Expected digests are recomputed from the same transformers calls TRL makes, never copied.
"""

from __future__ import annotations

import hashlib
from typing import Any

from vramforge_estimator.preprocessing import get_adapter
from vramforge_estimator.preprocessing.trl_common import token_digest
from vramforge_estimator.schemas import ColumnMapping, Objective

PREF = ColumnMapping(system="system", prompt="question", chosen="chosen", rejected="rejected")
ROW: dict[str, Any] = {
    "system": "",
    "question": "C 함수 하나 작성",
    "chosen": "```c\nint f(void){return 0;}\n```",
    "rejected": "몰라요",
    "lang": "c",
}
# The generation prompt ends with "x" that the full conversation never renders, so the prompt
# ids are not a prefix of the prompt+completion ids (TRL warns and slices anyway).
MISALIGNED = (
    "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
    "{% if add_generation_prompt %}assistant: x{% endif %}"
)


def ids(tok: Any, messages: list[dict[str, Any]], *, gen: bool = False) -> list[int]:
    return tok.apply_chat_template(messages, add_generation_prompt=gen)["input_ids"]


def u(text: str) -> dict[str, str]:
    return {"role": "user", "content": text}


def a(text: str) -> dict[str, str]:
    return {"role": "assistant", "content": text}


def test_encoding_is_documented_and_order_independent() -> None:
    single = hashlib.sha256(b"input_ids:1,2,3\n").hexdigest()
    assert token_digest({"input_ids": [1, 2, 3]}) == single
    pair = {"rejected_input_ids": [7], "chosen_input_ids": [5, 6]}
    expected = hashlib.sha256(b"chosen_input_ids:5,6\nrejected_input_ids:7\n").hexdigest()
    assert token_digest(pair) == expected == token_digest(dict(reversed(pair.items())))
    assert token_digest({"input_ids": [12, 3]}) != token_digest({"input_ids": [1, 23]})


def test_sft_digest_hashes_the_full_input_ids(mimo) -> None:
    adapter = get_adapter(Objective.SFT, mimo, PREF)
    rec = adapter.process(ROW, "train:0")
    full = ids(mimo.tokenizer, [u(ROW["question"]), a(ROW["chosen"])])
    assert rec.extras["token_digest"] == token_digest({"input_ids": full})
    # metadata and the unused rejected branch do not change what the model sees
    same = adapter.process({**ROW, "lang": "go", "rejected": "x"}, "train:1")
    assert same.extras["token_digest"] == rec.extras["token_digest"]
    other = adapter.process({**ROW, "chosen": ROW["chosen"] + " "}, "train:2")
    assert other.extras["token_digest"] != rec.extras["token_digest"]


def test_sft_text_digest_includes_the_appended_eos(plain) -> None:
    rec = get_adapter(Objective.SFT, plain, ColumnMapping(text="t")).process({"t": "hi"}, "r")
    expected = plain.tokenizer(text="hi</s>")["input_ids"]
    assert rec.extras["token_digest"] == token_digest({"input_ids": expected})


def test_dpo_digest_covers_both_branches_as_the_collator_builds_them(mimo) -> None:
    rec = get_adapter(Objective.DPO, mimo, PREF).process(ROW, "train:0")
    tok = mimo.tokenizer
    p = ids(tok, [u(ROW["question"])], gen=True)
    pc = ids(tok, [u(ROW["question"]), a(ROW["chosen"])])
    pr = ids(tok, [u(ROW["question"]), a(ROW["rejected"])])
    assert rec.extras["token_digest"] == token_digest(
        {"chosen_input_ids": p + pc[len(p) :], "rejected_input_ids": p + pr[len(p) :]}
    )


def test_dpo_digest_follows_trl_slicing_when_the_prompt_boundary_moves(make_handle) -> None:
    handle = make_handle("mimo_bytelevel", chat_template=MISALIGNED)
    rec = get_adapter(Objective.DPO, handle, PREF).process(ROW, "train:0")
    tok = handle.tokenizer
    p = ids(tok, [u(ROW["question"])], gen=True)
    pc = ids(tok, [u(ROW["question"]), a(ROW["chosen"])])
    pr = ids(tok, [u(ROW["question"]), a(ROW["rejected"])])
    assert pc[: len(p)] != p and rec.extras["prefix_mismatch"] is True
    sliced = {"chosen_input_ids": p + pc[len(p) :], "rejected_input_ids": p + pr[len(p) :]}
    assert rec.extras["token_digest"] == token_digest(sliced)
    rendered = {"chosen_input_ids": pc, "rejected_input_ids": pr}
    assert rec.extras["token_digest"] != token_digest(rendered)


def test_grpo_digest_hashes_the_generation_prompt(mimo) -> None:
    rec = get_adapter(Objective.GRPO, mimo, PREF).process(ROW, "train:0")
    prompt = ids(mimo.tokenizer, [u(ROW["question"])], gen=True)
    assert rec.extras["token_digest"] == token_digest({"prompt_ids": prompt})
    thinking_off = get_adapter(
        Objective.GRPO, mimo, PREF, template_kwargs={"enable_thinking": False}
    ).process(ROW, "train:0")
    assert thinking_off.extras["token_digest"] != rec.extras["token_digest"]


def test_failed_rows_have_no_digest(mimo) -> None:
    rec = get_adapter(Objective.SFT, mimo, PREF).process({**ROW, "chosen": None}, "train:0")
    assert not rec.ok and "token_digest" not in rec.extras
