"""Builders for model-inspection tests: the MiMo fixture as a local directory or a fake HF repo."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from vramforge_estimator.schemas import ModelSourceRef
from vramforge_estimator.sources import ResolvedSource, SourceAccess, hub, resolve_model

FIXTURES = Path(__file__).parents[2] / "fixtures"
MIMO = FIXTURES / "models" / "mimo-v2.6-distill-qwen-9b"
MIMO_REPO = "XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B"
MIMO_SHA = "2367e865d009c13ac81713a2878291d33ab28177"
# Published shard sizes and LFS sha256 (PROVENANCE.md)
MIMO_SHARDS = {
    "model-00001-of-00004.safetensors": (
        5276436216,
        "aab052180118aee34abc3029b54eaa49096aac606b97d703866b420dceb703c3",
    ),
    "model-00002-of-00004.safetensors": (
        5033171280,
        "7a0486565f06d25ac4628e9dba470dc3f604353471d240d5a0bf7128f64df396",
    ),
    "model-00003-of-00004.safetensors": (
        5234499504,
        "6c73207563d1879bfd6c143a028cc70be68edff56280458c71a43df4240300f4",
    ),
    "model-00004-of-00004.safetensors": (
        3275613848,
        "1379a7cf8c0b8555a39ab65a47e830e0eb45e776b46045734da3afd73c09eea2",
    ),
}


def load_fixture_module(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"vf_fixture_{name}", FIXTURES / "models" / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    if spec.name in sys.modules:
        return sys.modules[spec.name]
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def st_writer() -> ModuleType:
    return load_fixture_module("safetensors_writer")


@pytest.fixture(scope="session")
def fake_hub() -> ModuleType:
    return load_fixture_module("fake_hub")


def padded_header(raw: bytes) -> bytes:
    """Restore the 8-byte alignment padding removed from the fixture headers."""
    return raw + b" " * (-len(raw) % 8)


def mimo_headers() -> dict[str, tuple[bytes, dict[str, Any]]]:
    out = {}
    for shard in MIMO_SHARDS:
        raw = padded_header((MIMO / f"{shard}.header.json").read_bytes())
        out[shard] = (raw, json.loads(raw))
    return out


def write_mimo_dir(target: Path, st_writer: ModuleType) -> Path:
    """config + index + four shards with real header framing and sparse zero payloads."""
    target.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "model.safetensors.index.json"):
        (target / name).write_bytes((MIMO / name).read_bytes())
    for shard, (raw, header) in mimo_headers().items():
        size = st_writer.write_header(target / shard, header, raw=raw)
        assert size == MIMO_SHARDS[shard][0]
    return target


@pytest.fixture
def mimo_local(tmp_path: Path, st_writer: ModuleType) -> tuple[ResolvedSource, SourceAccess]:
    root = tmp_path / "models"
    write_mimo_dir(root / "mimo", st_writer)
    access = SourceAccess(local_roots={"models": root})
    return resolve_model(ModelSourceRef(reference="local:models/mimo"), access), access


def mimo_fake_repo(fake_hub: ModuleType) -> Any:
    return fake_hub.FakeRepo(
        repo_id=MIMO_REPO,
        sha=MIMO_SHA,
        files={
            "config.json": (MIMO / "config.json").read_bytes(),
            "model.safetensors.index.json": (MIMO / "model.safetensors.index.json").read_bytes(),
        },
        shards={
            shard: (header, MIMO_SHARDS[shard][0], MIMO_SHARDS[shard][1])
            for shard, (_, header) in mimo_headers().items()
        },
    )


@pytest.fixture
def mimo_hub(
    tmp_path: Path, fake_hub: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> tuple[ResolvedSource, SourceAccess, Any]:
    client = fake_hub.FakeHub(tmp_path / "hub", {("model", MIMO_REPO): mimo_fake_repo(fake_hub)})
    monkeypatch.setattr(hub, "get_hub_client", lambda access: client)
    access = SourceAccess(hf_home=tmp_path / "hf")
    source = resolve_model(ModelSourceRef(reference=MIMO_REPO, revision=MIMO_SHA), access)
    return source, access, client


def frame_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        prefix = handle.read(8)
        return hashlib.sha256(prefix + handle.read(int.from_bytes(prefix, "little"))).hexdigest()


@pytest.fixture(scope="session")
def mimo() -> SimpleNamespace:
    """Constants and helpers of the MiMo fixture for test modules."""
    return SimpleNamespace(
        dir=MIMO,
        repo=MIMO_REPO,
        sha=MIMO_SHA,
        shards=MIMO_SHARDS,
        write_dir=write_mimo_dir,
        frame_sha256=frame_sha256,
        fake_repo=mimo_fake_repo,
    )
