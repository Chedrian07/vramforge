"""Re-create this fixture from the Hugging Face Hub (network; never downloads weights).

    uv run --no-sync python tests/fixtures/models/mimo-v2.6-distill-qwen-9b/fetch_fixture.py

Writes config.json and model.safetensors.index.json as published, and for every shard only the
safetensors header: two ranged GETs (8-byte length prefix, then the JSON header). The header JSON
is stored without its trailing space padding as ``<shard>.header.json``. Prints the values
recorded in PROVENANCE.md.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download, hf_hub_url
from huggingface_hub.utils import build_hf_headers, get_session

REPO = "XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B"
REVISION = "2367e865d009c13ac81713a2878291d33ab28177"
HERE = Path(__file__).parent


def _range(url: str, start: int, end: int) -> bytes:
    response = get_session().get(
        url,
        headers={**build_hf_headers(token=False), "range": f"bytes={start}-{end}"},
        follow_redirects=True,
        timeout=30,
    )
    response.raise_for_status()
    return response.content


def main() -> None:
    for name in ("config.json", "model.safetensors.index.json"):
        path = Path(hf_hub_download(REPO, name, revision=REVISION, token=False))
        (HERE / name).write_bytes(path.read_bytes())
    index = json.loads((HERE / "model.safetensors.index.json").read_text(encoding="utf-8"))
    for shard in sorted(set(index["weight_map"].values())):
        url = hf_hub_url(REPO, shard, revision=REVISION)
        prefix = _range(url, 0, 7)
        length = int.from_bytes(prefix, "little")
        header = _range(url, 8, 8 + length - 1)
        assert len(header) == length
        compact = header.rstrip(b" ")
        json.loads(compact)
        (HERE / f"{shard}.header.json").write_bytes(compact)
        print(
            shard,
            "header_length",
            length,
            "frame_sha256",
            hashlib.sha256(prefix + header).hexdigest(),
            "fixture_sha256",
            hashlib.sha256(compact).hexdigest(),
        )
    for path in sorted(HERE.glob("*.json")):
        print(path.name, path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
