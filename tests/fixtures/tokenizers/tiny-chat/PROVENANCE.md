# Fixture provenance: tiny chat tokenizer

Synthetic tokenizer built offline with the `tokenizers` library by `build_fixture.py` in this
directory (deterministic with tokenizers==0.23.2; not taken from any model). Byte-level BPE
(vocab 384 incl. `<|endoftext|>`, `<|im_start|>`, `<|im_end|>`) plus the non-special added tokens
`<think>` and `</think>` → `len(tokenizer)` = 386. The ChatML-style template wraps assistant turns
in `{% generation %}` and reads `enable_thinking` for the generation prompt.

| File | Bytes | SHA256 |
|---|---:|---|
| `tokenizer.json` | 6,057 | `0c6e3a3520fc6f70906a5a38ca9983b8034ce0ac57324189dfa1e5e3cc8c1e8e` |
| `tokenizer_config.json` | 223 | `e0bd31f8b43a38882462e32700399976462775a2128982de2d43d20573e3b257` |
| `chat_template.jinja` | 535 | `f527a96140fe38e00e4bc7bb59d634d9a652fbc078f640198dc4ecda351b9da0` |

Used together with `tests/fixtures/models/tiny-dense-decoder` (vocab 512) by
`tests/unit/inspection_model/test_model_tokenizer.py`.
