"""Rebuild this synthetic tokenizer fixture (offline, deterministic for tokenizers==0.23.2).

    uv run --no-sync python tests/fixtures/tokenizers/tiny-chat/build_fixture.py

A byte-level BPE tokenizer trained on a few lines below, the ChatML-style special tokens, two
non-special think tokens, a tokenizer_config.json and a chat template with ``{% generation %}``
markers and an ``enable_thinking`` switch. Not taken from any model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers, trainers

HERE = Path(__file__).parent
SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
CORPUS = [
    "You are a helpful assistant.",
    "Write a function that adds two numbers.",
    "def add(a, b):\n    return a + b",
    "The quick brown fox jumps over the lazy dog.",
    "보안 취약점을 설명해 주세요.",
    "SELECT name FROM users WHERE id = 1;",
    "user assistant system tool",
] * 4

TOKENIZER_CONFIG = {
    "tokenizer_class": "PreTrainedTokenizerFast",
    "bos_token": None,
    "eos_token": "<|im_end|>",
    "pad_token": "<|endoftext|>",
    "unk_token": None,
    "model_max_length": 4096,
    "clean_up_tokenization_spaces": False,
}

TEMPLATE = """{%- for message in messages -%}
    {%- if message.role == 'assistant' -%}
        {%- generation -%}
        {{- '<|im_start|>assistant\\n<think></think>' ~ message.content ~ '<|im_end|>' -}}
        {%- endgeneration -%}
    {%- else -%}
        {{- '<|im_start|>' ~ message.role ~ '\\n' ~ message.content ~ '<|im_end|>' -}}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{- '<|im_start|>assistant\\n' -}}
    {%- if enable_thinking is false -%}
        {{- '<think></think>' -}}
    {%- endif -%}
{%- endif -%}
"""


def main() -> None:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=384,
        special_tokens=SPECIAL,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(CORPUS, trainer)
    tokenizer.add_tokens(
        [AddedToken("<think>", special=False), AddedToken("</think>", special=False)]
    )
    tokenizer.save(str(HERE / "tokenizer.json"), pretty=False)
    (HERE / "tokenizer_config.json").write_text(
        json.dumps(TOKENIZER_CONFIG, indent=2) + "\n", encoding="utf-8"
    )
    (HERE / "chat_template.jinja").write_text(TEMPLATE, encoding="utf-8")
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        data = (HERE / name).read_bytes()
        print(name, len(data), hashlib.sha256(data).hexdigest())


if __name__ == "__main__":
    main()
