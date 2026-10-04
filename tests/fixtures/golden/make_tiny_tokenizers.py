"""Regenerate the tiny offline tokenizer fixtures under tests/fixtures/golden/tokenizers/.

Run: ``uv run --no-sync python tests/fixtures/golden/make_tiny_tokenizers.py``

All three are byte-level BPE tokenizers without merges (one token per UTF-8 byte), so Korean,
code and long strings tokenize deterministically and fast without any download
(docs/research/trl-sft-dpo.md §11.1):

* ``mimo_bytelevel``  - MiMo special tokens + the real MiMo chat template (mimo_chat_template.jinja)
* ``bos_bytelevel``   - post-processor prepends ``<s>`` (BOS) and the template renders
  ``{{ bos_token }}`` too: the chat path must not duplicate BOS (plan §19.2)
* ``plain_bytelevel`` - no chat template: TRL's non-conversational path (EOS string appended)
"""

from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

HERE = Path(__file__).parent / "tokenizers"

BOS_TEMPLATE = (
    "{{ bos_token }}{% for m in messages %}[{{ m['role'] }}]{{ m['content'] }}"
    "{% if m['role'] == 'assistant' %}{{ eos_token }}{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}[assistant]{% endif %}"
)


def _core() -> Tokenizer:
    vocab = {ch: i for i, ch in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    core = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    core.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    core.decoder = decoders.ByteLevel()
    return core


def build_mimo() -> PreTrainedTokenizerFast:
    tok = PreTrainedTokenizerFast(
        tokenizer_object=_core(), eos_token="<|im_end|>", pad_token="<|endoftext|>"
    )
    tok.add_tokens(["<|im_start|>", "<think>", "</think>"])
    tok.chat_template = (HERE / "mimo_chat_template.jinja").read_text(encoding="utf-8")
    return tok


def build_bos() -> PreTrainedTokenizerFast:
    tok = PreTrainedTokenizerFast(
        tokenizer_object=_core(), bos_token="<s>", eos_token="</s>", pad_token="<pad>"
    )
    bos_id = tok.convert_tokens_to_ids("<s>")
    tok._tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A", pair="<s> $A <s> $B", special_tokens=[("<s>", bos_id)]
    )
    tok.chat_template = BOS_TEMPLATE
    return tok


def build_plain() -> PreTrainedTokenizerFast:
    return PreTrainedTokenizerFast(tokenizer_object=_core(), eos_token="</s>", pad_token="<pad>")


def main() -> None:
    for name, builder in (
        ("mimo_bytelevel", build_mimo),
        ("bos_bytelevel", build_bos),
        ("plain_bytelevel", build_plain),
    ):
        builder().save_pretrained(HERE / name)


if __name__ == "__main__":
    main()
