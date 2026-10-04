# Fixture provenance: tiny synthetic dense decoder

Synthetic, hand-written fixture (not taken from any repository). It exercises the dense-decoder
path of the model inspector: no `head_dim` (inferred), no `layer_types` (all full attention),
tied embeddings without a serialized `lm_head`, and the legacy `torch_dtype` key.

| File | Bytes | SHA256 |
|---|---:|---|
| `config.json` | 396 | `8f28777b6873fca93b60e8c3a103f163cf5d4672696ec3b2ad5019020c2ac346` |
| `model.safetensors.header.json` | 2,039 | `ebf78fb46a4592931f95f01b65b751939330b51ee77f3d740728bf420004694f` |

`model.safetensors.header.json` was produced with
`tests/fixtures/models/safetensors_writer.py::header_for` (contiguous `data_offsets` in insertion
order, `__metadata__ = {"format": "pt"}`) for this Llama-style layout, F16:

- `model.embed_tokens.weight` `[256, 64]`
- per layer (2 layers): `input_layernorm` `[64]`, `self_attn.{q,o}_proj` `[64, 64]`,
  `self_attn.{k,v}_proj` `[32, 64]`, `post_attention_layernorm` `[64]`,
  `mlp.{gate,up}_proj` `[128, 64]`, `mlp.down_proj` `[64, 128]`
- `model.norm.weight` `[64]`

Expected inventory: 20 tensors, 90,432 parameters (180,864 bytes), 14 Linear modules
(8 attention + 6 MLP), tied group `model.embed_tokens.weight` ↔ `lm_head.weight`, head_dim 16
(inferred from 64 / 4).
