# Model inventory fixtures

Offline fixtures for the architecture adapters (`packages/estimator/.../architectures/`).
Tests never touch the network; the headers below were fetched once.

| File | Content |
|---|---|
| `mimo-v2.6-distill-qwen-9b.headers.json` | Pinned `config.json` + one `[name, dtype, shape, shard]` row per tensor (760 rows) + source identity |
| `fetch_mimo_headers.py` | The fetch script (safetensors **headers only**, no weights) |
| `inventory_builder.py` | Builds a `ModelInventory` from config + tensor rows; synthetic tiny Qwen3.5 / dense configs |

## Source

| Field | Value |
|---|---|
| Repo | `XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B` |
| Revision | `2367e865d009c13ac81713a2878291d33ab28177` |
| `config.json` sha256 | `407c46388b8fa2ae9bf69fe27d40af236d373e86a6b48f5284c86df5cd183633` |
| `model.safetensors.index.json` sha256 | `4a24d25e902169c0393a2ca58b697c03b27efc40dcddbc2b59c42e5e1e2b1d90` |
| Shard LFS sha256 / size | recorded in the JSON `source.shards` |
| Parameters (header sum) | 9,409,813,744 (text 8,953,803,264 + vision 456,010,480), all BF16 |
| Fetched | 2026-10-04 with `huggingface_hub==1.33.0` `get_safetensors_metadata` |

These numbers match `docs/research/loading-quantization-peft.md` (V1) and
`docs/research/architecture-memory.md` §9.

## Regenerate

```bash
HF_HOME=/tmp/vf-research/hf uv run --no-sync python tests/fixtures/inventories/fetch_mimo_headers.py
```

Review the diff: a changed revision must also update the expected values in
`tests/unit/architectures/` (4-bit bytes, LoRA counts, cache bytes).

## Builder conventions (what the adapters expect from an inventory)

- `component`: `model.visual.*` → `vision`, `mtp.*` → `mtp`, everything else → `text`.
- `role`: `embedding` (embed_tokens, vision pos_embed), `lm_head`, `norm` (module leaf contains
  `norm`), `conv` (conv1d, patch_embed), `parameter` (`A_log`, `dt_bias`), `linear_weight` /
  `linear_bias` for 2-D Linear modules.
- `linear_modules`: one entry per 2-D Linear (`kind` = module leaf, `in/out_features` from the
  weight shape); `lm_head` is included with kind `lm_head` (the adapters also accept inventories
  without it). Text entries carry `layer_index`.
- `tied_groups`: `[embed_tokens, lm_head]` when the config ties them.

Usage from a test (there are no `__init__.py` files under `tests/`):

```python
import importlib.util

spec = importlib.util.spec_from_file_location("inventory_builder", PATH_TO / "inventory_builder.py")
```
