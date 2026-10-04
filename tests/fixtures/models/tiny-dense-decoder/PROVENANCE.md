# Fixture provenance: tiny synthetic dense decoder

Synthetic, hand-written fixture (not taken from any repository). It exercises the dense-decoder
path of the model inspector: no `head_dim` (inferred), no `layer_types` (all full attention),
tied embeddings without a serialized `lm_head`, and the legacy `torch_dtype` key.

| File | Bytes | SHA256 |
|---|---:|---|
| `config.json` | 396 | `286b6bf00ba7b1a8615dc057311f39e71c142e3fdbcb3e181d898f38deb95fd4` |
| `model.safetensors.header.json` | 2,043 | `37992eef728ad7d3a462ce70a081bcddadfd71c074b9d1f33dc307cfa5f69bb5` |

`model.safetensors.header.json` was produced with
`tests/fixtures/models/safetensors_writer.py::header_for` (contiguous `data_offsets` in insertion
order, `__metadata__ = {"format": "pt"}`) for this Llama-style layout, F16:

- `model.embed_tokens.weight` `[512, 64]` (vocab 512 ≥ the 386 tokens of
  `tests/fixtures/tokenizers/tiny-chat`, like a real checkpoint)
- per layer (2 layers): `input_layernorm` `[64]`, `self_attn.{q,o}_proj` `[64, 64]`,
  `self_attn.{k,v}_proj` `[32, 64]`, `post_attention_layernorm` `[64]`,
  `mlp.{gate,up}_proj` `[128, 64]`, `mlp.down_proj` `[64, 128]`
- `model.norm.weight` `[64]`

Expected inventory: 20 tensors, 106,816 parameters (213,632 bytes), 14 Linear modules
(8 attention + 6 MLP), tied group `model.embed_tokens.weight` ↔ `lm_head.weight`, head_dim 16
(inferred from 64 / 4).
