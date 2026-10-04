# Fixture provenance: MiMo-V2.6-Distill-Qwen-9B metadata

| Field | Value |
|---|---|
| Repository | `XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B` (model card license: MIT) |
| Revision | `2367e865d009c13ac81713a2878291d33ab28177` |
| Retrieved | 2026-10-04 |
| Method | `fetch_fixture.py` in this directory: `hf_hub_download` for `config.json` and `model.safetensors.index.json`; for each shard two ranged GETs (bytes 0-7 = header length, then the header). Weights were never downloaded. |
| Used by | `tests/unit/inspection_model/` (inventory of the plan's example model, plan.md §6.4, §21) |

The `*.header.json` files are the shard headers exactly as published, with the trailing space
padding removed (padding brings the header length to a multiple of 8). Tests rebuild real
safetensors framing from them: `length = header_length` below, padding = spaces, payload = zeros
(sparse) up to the published shard size.

## Files

| File | Bytes | SHA256 | Hub reference |
|---|---:|---|---|
| `config.json` | 2,784 | `407c46388b8fa2ae9bf69fe27d40af236d373e86a6b48f5284c86df5cd183633` | git blob `007c2eb8c73b9295f22491b1e4ab29d8b10f6e05` |
| `model.safetensors.index.json` | 69,217 | `4a24d25e902169c0393a2ca58b697c03b27efc40dcddbc2b59c42e5e1e2b1d90` | git blob `be2a44ec6dc7004e51e648b5e2cf8c32cee42c79` |
| `model-00001-of-00004.safetensors.header.json` | 1,774 | `35fc1b64c3265e6cfaff2c913195ee9ddb4f9daceed5a5df613ead548cb917bf` | see shard table |
| `model-00002-of-00004.safetensors.header.json` | 6,472 | `85e091943c2a95b0d89e0be76a7db27ab221e470081ec6fc16419746015033e5` | see shard table |
| `model-00003-of-00004.safetensors.header.json` | 8,100 | `0bc2e3d6f00c0202c51b767c3fe2e1ec7429ebd6c98e7090a07a823d73277bfa` | see shard table |
| `model-00004-of-00004.safetensors.header.json` | 76,970 | `4852c42fe4c8353786309844db433f3bbba307b3b243cf8b040db636d360d4be` | see shard table |

## Shards

`header_length` is the little-endian u64 at offset 0 of the published file; `frame sha256` is the
sha256 of the first `8 + header_length` bytes (prefix + header incl. padding).

| Shard | Published size | LFS sha256 | header_length | Frame sha256 |
|---|---:|---|---:|---|
| `model-00001-of-00004.safetensors` | 5,276,436,216 | `aab052180118aee34abc3029b54eaa49096aac606b97d703866b420dceb703c3` | 1,776 | `b9a4eb7e269fed6f83f046413693444f83581583ef7d10fcdd2123174295340b` |
| `model-00002-of-00004.safetensors` | 5,033,171,280 | `7a0486565f06d25ac4628e9dba470dc3f604353471d240d5a0bf7128f64df396` | 6,472 | `4446e06139df99a4b1456b52cafc72ccac28035d82fe2d6baef695d580a6d6b1` |
| `model-00003-of-00004.safetensors` | 5,234,499,504 | `6c73207563d1879bfd6c143a028cc70be68edff56280458c71a43df4240300f4` | 8,104 | `0bdfb937253ee825744db5e76e3925bd013f0d6af2c3bbfda69d16d4f69afad1` |
| `model-00004-of-00004.safetensors` | 3,275,613,848 | `1379a7cf8c0b8555a39ab65a47e830e0eb45e776b46045734da3afd73c09eea2` | 76,976 | `7d2c7f4b2eaa619eb6b539a018adb78c3daa826a34e68548a2738625c5a0041e` |

## Facts the tests check (cross-checked with docs/research)

- 760 tensors, all BF16, 9,409,813,744 parameters; `model.visual.*` 456,010,480; the rest
  8,953,803,264 (docs/research/architecture-memory.md §9, docs/research/loading-quantization-peft.md V1).
- `index.metadata.total_size` = 18,819,627,488 = 2 × parameters.
- Text layers: 24 `linear_attention` + 8 `full_attention`; `head_dim` 256 set explicitly;
  `tie_word_embeddings` false at both levels; no `mtp.*` tensors although the config declares
  `mtp_num_hidden_layers: 1` (docs/research/example-model-dataset.md O7).
- Linear (bnb-convertible) modules: 248 text + 110 vision (docs/research/loading-quantization-peft.md §2.4).
