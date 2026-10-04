"""Fetch the safetensors HEADERS (never the weights) of the example model into a compact fixture.

Run once with network access (tests never call this):

    HF_HOME=/tmp/vf-research/hf uv run --no-sync python \
        tests/fixtures/inventories/fetch_mimo_headers.py

Output: `mimo-v2.6-distill-qwen-9b.headers.json` next to this script. It holds the pinned config.json,
one `[name, safetensors dtype, shape, shard]` row per tensor and the source identity (repo,
revision, sha256 of config/index, LFS sha256 + size of every shard). `inventory_builder.py` turns
it into a `ModelInventory` at test time.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, get_safetensors_metadata, hf_hub_download

REPO_ID = "XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B"
REVISION = "2367e865d009c13ac81713a2878291d33ab28177"
OUT = Path(__file__).with_name("mimo-v2.6-distill-qwen-9b.headers.json")


def _sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> None:
    config_path = hf_hub_download(REPO_ID, "config.json", revision=REVISION)
    index_path = hf_hub_download(REPO_ID, "model.safetensors.index.json", revision=REVISION)
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))

    meta = get_safetensors_metadata(REPO_ID, revision=REVISION)
    rows: list[list[object]] = []
    for shard, file_meta in sorted(meta.files_metadata.items()):
        for name, info in file_meta.tensors.items():
            rows.append([name, info.dtype, list(info.shape), shard])
    rows.sort(key=lambda r: str(r[0]))

    info = HfApi().model_info(REPO_ID, revision=REVISION, files_metadata=True)
    shards = {
        s.rfilename: {"lfs_sha256": s.lfs.sha256 if s.lfs else None, "size": s.size}
        for s in info.siblings or []
        if s.rfilename.endswith(".safetensors")
    }

    doc = {
        "source": {
            "repo_id": REPO_ID,
            "revision": REVISION,
            "method": "huggingface_hub.get_safetensors_metadata (headers only, no weights)",
            "config_sha256": _sha256(config_path),
            "index_sha256": _sha256(index_path),
            "parameter_count": sum(meta.parameter_count.values()),
            "shards": shards,
        },
        "config": config,
        "tensors": rows,
    }
    # One tensor per line keeps the fixture reviewable in diffs.
    head = json.dumps({k: v for k, v in doc.items() if k != "tensors"}, indent=1, sort_keys=True)
    body = ",\n".join("  " + json.dumps(r, separators=(",", ":")) for r in rows)
    OUT.write_text(head[:-2] + ',\n "tensors": [\n' + body + "\n ]\n}\n", encoding="utf-8")
    print(f"wrote {OUT.name}: {len(rows)} tensors")


if __name__ == "__main__":
    main()
