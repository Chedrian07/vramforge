"""SFT adapter = TRL 1.14.1 SFT preprocessing with max_length=None (trl-sft-dpo.md §2).

Expected values are computed with the same transformers calls TRL makes, never hand-copied.
"""

from __future__ import annotations

from typing import Any

import pytest

from vramforge_estimator.preprocessing import get_adapter
from vramforge_estimator.preprocessing.trl_sft import TrlSftAdapter
from vramforge_estimator.schemas import ColumnMapping, EmptySystemPolicy, ErrorCode, Objective

PREF = ColumnMapping(system="system", prompt="question", chosen="chosen", rejected="rejected")
ROW: dict[str, Any] = {
    "system": "",
    "question": "파이썬 eval 위험성을 설명해줘",
    "chosen": "eval은 위험합니다.\n```python\nimport ast\nast.literal_eval(x)\n```",
    "rejected": "괜찮아요",
    "lang": "python",
    "vulnerability": "metadata that must never be rendered",
}


def chat_ids(
    tok: Any, messages: list[dict[str, Any]], *, gen: bool = False, **kw: Any
) -> list[int]:
    return tok.apply_chat_template(messages, add_generation_prompt=gen, **kw)["input_ids"]


def user(text: str) -> dict[str, str]:
    return {"role": "user", "content": text}


def assistant(text: str) -> dict[str, str]:
    return {"role": "assistant", "content": text}


def test_conversational_prompt_completion_matches_trl(mimo) -> None:
    rec = get_adapter(Objective.SFT, mimo, PREF).process(ROW, "train:0")
    tok = mimo.tokenizer
    prompt_ids = chat_ids(tok, [user(ROW["question"])], gen=True)
    full_ids = chat_ids(tok, [user(ROW["question"]), assistant(ROW["chosen"])])
    assert rec.ok and rec.error_code is None
    assert rec.prompt_tokens == len(prompt_ids)
    assert rec.sequence_tokens == len(full_ids)
    assert rec.completion_tokens == len(full_ids) - len(prompt_ids)
    # MiMo layout: <think></think> + content + <|im_end|> (example-model-dataset.md §5.4)
    assert rec.completion_tokens == len(ROW["chosen"].encode()) + 3
    assert rec.template_preserved is True and rec.template_issue is None
    assert set(rec.extras) == {"system_omitted", "token_digest"}
    assert rec.extras["system_omitted"] == 1


def test_prompt_masked_tokens_stay_in_the_sequence_length(mimo) -> None:
    rec = get_adapter(Objective.SFT, mimo, PREF).process(ROW, "train:0")
    assert rec.sequence_tokens == rec.prompt_tokens + rec.completion_tokens
    assert rec.loss_token_count == rec.completion_tokens  # completion_only_loss
    assert rec.loss_token_count < rec.sequence_tokens
    assert rec.max_sequence_tokens() == rec.sequence_tokens


def test_metadata_columns_are_never_rendered(mimo) -> None:
    adapter = get_adapter(Objective.SFT, mimo, PREF)
    with_meta = adapter.process(ROW, "train:0")
    stripped = {k: ROW[k] for k in ("system", "question", "chosen", "rejected")}
    without = adapter.process(stripped, "train:0")
    assert with_meta.sequence_tokens == without.sequence_tokens
    assert with_meta.content_digest == without.content_digest
    # SFT trains on chosen only: rejected does not change the mapped record
    other_rejected = adapter.process({**ROW, "rejected": "다른 응답"}, "train:0")
    assert other_rejected.content_digest == with_meta.content_digest
    other_chosen = adapter.process({**ROW, "chosen": "다른 응답"}, "train:0")
    assert other_chosen.content_digest != with_meta.content_digest


def test_keep_policy_renders_the_empty_system_message(mimo) -> None:
    keep = PREF.model_copy(update={"empty_system_policy": EmptySystemPolicy.KEEP})
    omit_rec = get_adapter(Objective.SFT, mimo, PREF).process(ROW, "r")
    keep_rec = get_adapter(
        Objective.SFT, mimo, keep, empty_system_policy=EmptySystemPolicy.KEEP
    ).process(ROW, "r")
    tok = mimo.tokenizer
    system_block = len(chat_ids(tok, [{"role": "system", "content": ""}, user("q")])) - len(
        chat_ids(tok, [user("q")])
    )
    assert system_block > 0
    assert keep_rec.sequence_tokens == omit_rec.sequence_tokens + system_block
    assert keep_rec.prompt_tokens == omit_rec.prompt_tokens + system_block
    assert keep_rec.completion_tokens == omit_rec.completion_tokens
    assert keep_rec.content_digest != omit_rec.content_digest
    assert "system_omitted" not in keep_rec.extras


@pytest.mark.parametrize(("system", "rendered"), [("   \n", False), ("보안 리뷰어", True)])
def test_whitespace_system_is_omitted_but_real_system_is_kept(mimo, system, rendered) -> None:
    adapter = get_adapter(Objective.SFT, mimo, PREF)
    base = adapter.process(ROW, "r")
    rec = adapter.process({**ROW, "system": system}, "r")
    expected = chat_ids(
        mimo.tokenizer,
        ([{"role": "system", "content": system}] if rendered else [])
        + [user(ROW["question"]), assistant(ROW["chosen"])],
    )
    assert rec.sequence_tokens == len(expected)
    assert (rec.sequence_tokens > base.sequence_tokens) is rendered


def test_messages_layout_labels_every_token_but_the_first(mimo) -> None:
    conv = [user("2+2는?"), assistant("4"), user("코드로?"), assistant("```py\nprint(4)\n```")]
    rec = get_adapter(Objective.SFT, mimo, ColumnMapping(messages="conv")).process(
        {"conv": conv, "id": 7}, "train:3"
    )
    assert rec.ok
    assert rec.sequence_tokens == len(chat_ids(mimo.tokenizer, conv))
    assert rec.loss_token_count == rec.sequence_tokens - 1  # labels == input_ids, causal shift
    assert rec.prompt_tokens is None and rec.completion_tokens is None


def test_messages_assistant_only_loss_counts_the_generation_span(mimo) -> None:
    conv = [user("질문"), assistant("답변 ```x```")]
    adapter = TrlSftAdapter(mimo, ColumnMapping(messages="conv"), assistant_only_loss=True)
    rec = adapter.process({"conv": conv}, "r")
    out = mimo.tokenizer.apply_chat_template(
        conv, return_dict=True, return_assistant_tokens_mask=True
    )
    assert rec.loss_token_count == sum(out["assistant_masks"][1:])
    # MiMo's {% generation %} wraps the whole assistant turn, header included
    assert rec.loss_token_count > len("답변 ```x```".encode()) + 3


def test_prompt_completion_assistant_only_loss_equals_completion(mimo) -> None:
    adapter = TrlSftAdapter(mimo, PREF, assistant_only_loss=True)
    rec = adapter.process(ROW, "r")
    plain = get_adapter(Objective.SFT, mimo, PREF).process(ROW, "r")
    assert rec.loss_token_count == plain.loss_token_count == plain.completion_tokens


def test_text_layout_appends_eos_once(plain) -> None:
    adapter = get_adapter(Objective.SFT, plain, ColumnMapping(text="text"))
    tok = plain.tokenizer
    rec = adapter.process({"text": "hello 세계"}, "r")
    assert rec.sequence_tokens == len(tok(text="hello 세계</s>")["input_ids"])
    assert rec.sequence_tokens == len("hello 세계".encode()) + 1
    already = adapter.process({"text": "hello 세계</s>"}, "r")
    assert already.sequence_tokens == rec.sequence_tokens
    assert "duplicate_eos" not in already.extras
    assert rec.loss_token_count == rec.sequence_tokens - 1


def test_plain_prompt_completion_without_template(plain) -> None:
    mapping = ColumnMapping(prompt="p", completion="c")
    rec = get_adapter(Objective.SFT, plain, mapping).process({"p": "Q: 2+2?", "c": " 4"}, "r")
    tok = plain.tokenizer
    assert rec.prompt_tokens == len(tok(text="Q: 2+2?")["input_ids"])
    assert rec.sequence_tokens == len(tok(text="Q: 2+2? 4</s>")["input_ids"])
    assert rec.completion_tokens == rec.loss_token_count == len(b" 4") + 1
    assert rec.template_preserved is None


def test_plain_path_cannot_express_a_system_message(plain) -> None:
    mapping = ColumnMapping(system="s", prompt="p", completion="c")
    adapter = get_adapter(Objective.SFT, plain, mapping)
    assert adapter.process({"s": "", "p": "a", "c": "b"}, "r").ok  # empty -> omitted
    rec = adapter.process({"s": "be nice", "p": "a", "c": "b"}, "r")
    assert not rec.ok and rec.error_code is ErrorCode.TEMPLATE_REQUIRED


def test_chat_path_does_not_duplicate_bos(bos) -> None:
    mapping = ColumnMapping(prompt="p", completion="c")
    rec = get_adapter(Objective.SFT, bos, mapping).process({"p": "hi", "c": "yo"}, "r")
    tok = bos.tokenizer
    rendered = tok.apply_chat_template([user("hi"), assistant("yo")], tokenize=False)
    assert rendered.startswith("<s>")
    single = len(tok(rendered, add_special_tokens=False)["input_ids"])
    assert len(tok(rendered)["input_ids"]) == single + 1  # default call would add a 2nd BOS
    assert rec.sequence_tokens == single
    assert "duplicate_bos" not in rec.extras


def test_plain_text_starting_with_bos_string_is_flagged(bos) -> None:
    rec = get_adapter(Objective.SFT, bos, ColumnMapping(text="t")).process({"t": "<s>hi"}, "r")
    tok = bos.tokenizer
    # TRL tokenizes with add_special_tokens=True: BOS text + post-processor BOS -> reproduced
    assert rec.sequence_tokens == len(tok(text="<s>hi</s>")["input_ids"])
    assert rec.extras.get("duplicate_bos") is True


def test_long_strings_are_never_truncated(mimo) -> None:
    long_answer = "x" * 200_000
    rec = get_adapter(Objective.SFT, mimo, PREF).process({**ROW, "chosen": long_answer}, "r")
    assert rec.completion_tokens == 200_000 + 3
    assert rec.sequence_tokens == rec.prompt_tokens + 200_003


def test_template_content_loss_is_detected(make_handle) -> None:
    clipping = (
        "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'][:8] }}<|im_end|>"
        "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )
    handle = make_handle("mimo_bytelevel", chat_template=clipping)
    rec = get_adapter(Objective.SFT, handle, PREF).process(ROW, "r")
    assert rec.ok  # lengths are still exact; the loss is reported
    assert rec.template_preserved is False
    assert rec.template_issue and "assistant" in rec.template_issue
    assert ROW["chosen"] not in rec.template_issue


@pytest.mark.parametrize(
    ("row", "code"),
    [
        ({"system": "", "question": "q"}, ErrorCode.COLUMN_MAPPING_REQUIRED),
        ({"system": "", "question": None, "chosen": "a"}, ErrorCode.DATASET_FORMAT_UNSUPPORTED),
        ({"system": "", "question": "q", "chosen": 42}, ErrorCode.DATASET_FORMAT_UNSUPPORTED),
        ({"system": 3, "question": "q", "chosen": "a"}, ErrorCode.DATASET_FORMAT_UNSUPPORTED),
    ],
)
def test_bad_rows_are_returned_not_raised(mimo, row, code) -> None:
    rec = get_adapter(Objective.SFT, mimo, PREF).process(row, "train:9")
    assert not rec.ok and rec.error_code is code
    assert rec.row_id == "train:9" and rec.error_message
    assert rec.sequence_tokens is None


def test_non_text_content_blocks_are_rejected(mimo) -> None:
    conv = [{"role": "user", "content": [{"type": "image", "image": "x.png"}]}, assistant("a")]
    rec = get_adapter(Objective.SFT, mimo, ColumnMapping(messages="m")).process({"m": conv}, "r")
    assert not rec.ok and rec.error_code is ErrorCode.DATASET_FORMAT_UNSUPPORTED


def test_template_errors_are_reported_without_row_text(make_handle) -> None:
    raising = "{{ raise_exception('Conversation roles must alternate') }}"
    handle = make_handle("mimo_bytelevel", chat_template=raising)
    rec = get_adapter(Objective.SFT, handle, PREF).process(ROW, "r")
    assert not rec.ok and rec.error_code is ErrorCode.DATASET_FORMAT_UNSUPPORTED
    assert "Conversation roles must alternate" in rec.error_message
    assert ROW["question"] not in rec.error_message


@pytest.mark.parametrize(
    ("template", "secret"),
    [
        (  # a template that quotes a tool-call argument in its error message
            "{% for m in messages %}{% if m.tool_calls is defined %}{{ raise_exception("
            "'bad tool call: ' ~ m.tool_calls[0].function.arguments.query) }}{% endif %}"
            "{{ m.content }}{% endfor %}",
            "SECRET customer 4111-1111",
        ),
        ("{{ raise_exception('Unknown role: ' ~ messages[0].role) }}", "internal_audit_note"),
        ("{{ raise_exception('Bad content: ' ~ messages[0].content[:12]) }}", "confidential"),
    ],
)
def test_template_errors_built_from_row_values_are_not_echoed(
    make_handle, template, secret
) -> None:
    conv = [
        {"role": "internal_audit_note", "content": "confidential case notes"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"type": "function", "function": {"name": "s", "arguments": {"query": secret}}}
            ],
        },
    ]
    handle = make_handle("mimo_bytelevel", chat_template=template)
    rec = get_adapter(Objective.SFT, handle, ColumnMapping(messages="m")).process({"m": conv}, "r")
    assert not rec.ok and rec.error_code is ErrorCode.DATASET_FORMAT_UNSUPPORTED
    assert secret not in rec.error_message and "TemplateError" in rec.error_message


def test_template_kwargs_reach_the_template(mimo) -> None:
    adapter = get_adapter(Objective.SFT, mimo, PREF, template_kwargs={"enable_thinking": False})
    base = get_adapter(Objective.SFT, mimo, PREF).process(ROW, "r")
    rec = adapter.process(ROW, "r")
    # enable_thinking=False moves <think></think> into the generation prompt (+2 prompt tokens)
    assert rec.prompt_tokens == base.prompt_tokens + 2
    assert rec.sequence_tokens == base.sequence_tokens
    assert rec.completion_tokens == base.completion_tokens - 2
