"""Unsupported formats, pre-quantized checkpoints, remote code and integrity (plan.md §6.3, §6.5, §18)."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from huggingface_hub import errors as hf_errors

from vramforge_estimator.errors import EstimatorError
from vramforge_estimator.inspection import inspect_model
from vramforge_estimator.schemas import ErrorCode, ModelInventory, ModelSourceRef
from vramforge_estimator.sources import SourceAccess, hub, resolve_model

CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 8,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "vocab_size": 16,
    "tie_word_embeddings": False,
}
TENSORS: dict[str, tuple[str, list[int]]] = {
    "model.embed_tokens.weight": ("BF16", [16, 8]),
    "model.layers.0.self_attn.q_proj.weight": ("BF16", [8, 8]),
    "lm_head.weight": ("BF16", [16, 8]),
}

ModelDir = Callable[..., tuple[Path, SourceAccess]]


@pytest.fixture
def model_dir(tmp_path: Path, st_writer: Any) -> ModelDir:
    def make(
        files: dict[str, bytes] | None = None,
        *,
        config: dict[str, Any] | None = CONFIG,
        tensors: dict[str, tuple[str, list[int]]] | None = TENSORS,
        name: str = "m",
    ) -> tuple[Path, SourceAccess]:
        root = tmp_path / "models"
        target = root / name
        target.mkdir(parents=True)
        if config is not None:
            (target / "config.json").write_text(json.dumps(config), encoding="utf-8")
        if tensors is not None:
            st_writer.write_safetensors(target / "model.safetensors", tensors)
        for rel, content in (files or {}).items():
            (target / rel).parent.mkdir(parents=True, exist_ok=True)
            (target / rel).write_bytes(content)
        return target, SourceAccess(local_roots={"models": root})

    return make


def _inspect(target: Path, access: SourceAccess) -> ModelInventory:
    source = resolve_model(ModelSourceRef(reference=f"local:models/{target.name}"), access)
    return inspect_model(source, access)


def _fails(code: ErrorCode, target: Path, access: SourceAccess, **details: Any) -> EstimatorError:
    with pytest.raises(EstimatorError) as exc:
        _inspect(target, access)
    issue = exc.value.issue
    assert issue.code is code, issue
    assert issue.stage is not None and issue.stage.value == "inspecting"
    for key, value in details.items():
        assert issue.details.get(key) == value, issue.details
    assert str(target) not in issue.model_dump_json()
    return exc.value


def test_valid_minimal_model(model_dir: ModelDir) -> None:
    inv = _inspect(*model_dir())
    assert inv.params_total == 16 * 8 * 2 + 64
    assert [m.name for m in inv.linear_modules] == ["model.layers.0.self_attn.q_proj"]


@pytest.mark.parametrize(
    ("files", "fmt"),
    [
        ({"model-Q4_K_M.gguf": b"GGUF\x03\x00"}, "gguf"),
        (
            {"pytorch_model.bin": b"\x80\x04 pickle", "pytorch_model.bin.index.json": b"{}"},
            "pytorch_pickle",
        ),
        ({"model.pt": b"\x80\x04"}, "pickle"),
        ({"tf_model.h5": b"\x89HDF"}, "tensorflow"),
        ({"flax_model.msgpack": b"\x80"}, "flax"),
        ({"model.onnx": b"onnx"}, "onnx"),
        ({"consolidated.safetensors": b"x" * 16}, "nonstandard_safetensors"),
        ({"README.md": b"# no weights"}, "no_weights"),
        ({"adapter_config.json": b"{}", "adapter_model.safetensors": b"x" * 16}, "adapter_only"),
    ],
)
def test_unsupported_weight_formats(model_dir: ModelDir, files: dict[str, bytes], fmt: str) -> None:
    target, access = model_dir(files, tensors=None)
    _fails(ErrorCode.UNSUPPORTED_MODEL_FORMAT, target, access, format=fmt)


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read mode-000 files")
def test_pickle_weights_are_never_opened(model_dir: ModelDir) -> None:
    target, access = model_dir({"pytorch_model.bin": b"\x80\x04 malicious pickle"}, tensors=None)
    (target / "pytorch_model.bin").chmod(0)  # any attempt to read it would raise
    try:
        _fails(ErrorCode.UNSUPPORTED_MODEL_FORMAT, target, access, format="pytorch_pickle")
    finally:
        (target / "pytorch_model.bin").chmod(0o600)


def test_extra_formats_next_to_safetensors_are_ignored(model_dir: ModelDir) -> None:
    inv = _inspect(*model_dir({"model-Q4_K_M.gguf": b"GGUF", "pytorch_model.bin": b"\x80"}))
    assert inv.facts.extra["ignored_weight_formats"] == ["gguf", "pytorch_pickle"]


def test_declared_quantization_config(model_dir: ModelDir) -> None:
    config = {**CONFIG, "quantization_config": {"quant_method": "awq", "bits": 4}}
    _fails(
        ErrorCode.UNSUPPORTED_MODEL_FORMAT,
        *model_dir(config=config),
        reason="prequantized_checkpoint",
        quantized_checkpoint_format="awq",
    )


@pytest.mark.parametrize(
    ("tensors", "fmt"),
    [
        (
            {
                "model.layers.0.self_attn.q_proj.qweight": ("I32", [1, 8]),
                "model.layers.0.self_attn.q_proj.qzeros": ("I32", [1, 1]),
                "model.layers.0.self_attn.q_proj.scales": ("F16", [1, 8]),
            },
            "gptq/awq",
        ),
        ({"model.layers.0.mlp.up_proj.weight": ("F8_E4M3", [8, 8])}, "fp8"),
        ({"model.layers.0.mlp.up_proj.weight": ("U8", [8, 4])}, "integer-packed"),
        (
            {
                "model.layers.0.mlp.up_proj.weight": ("U8", [32, 1]),
                "model.layers.0.mlp.up_proj.weight.absmax": ("F32", [1]),
            },
            "bitsandbytes",
        ),
    ],
)
def test_prequantized_tensors_without_config(
    model_dir: ModelDir, tensors: dict[str, tuple[str, list[int]]], fmt: str
) -> None:
    _fails(
        ErrorCode.UNSUPPORTED_MODEL_FORMAT,
        *model_dir(tensors={**TENSORS, **tensors}),
        reason="prequantized_checkpoint",
        quantized_checkpoint_format=fmt,
    )


def test_remote_code_is_refused(model_dir: ModelDir) -> None:
    config = {
        **CONFIG,
        "model_type": "vramforge_custom_arch",
        "architectures": ["CustomForCausalLM"],
        "auto_map": {"AutoModelForCausalLM": "modeling_custom.CustomForCausalLM"},
    }
    target, access = model_dir(
        {"modeling_custom.py": b"raise SystemExit('never executed')"}, config=config
    )
    _fails(ErrorCode.REMOTE_CODE_REQUIRED, target, access, model_type="vramforge_custom_arch")


def test_auto_map_with_native_class_is_fine(model_dir: ModelDir) -> None:
    config = {**CONFIG, "auto_map": {"AutoModelForCausalLM": "modeling_x.X"}}
    inv = _inspect(*model_dir(config=config))
    assert inv.facts.extra["auto_map_present"] is True


def test_unknown_dtype(model_dir: ModelDir, st_writer: Any) -> None:
    target, access = model_dir(tensors=None)
    header = {"w.weight": {"dtype": "F8_E8M0", "shape": [4], "data_offsets": [0, 4]}}
    st_writer.write_header(target / "model.safetensors", header)
    _fails(
        ErrorCode.UNSUPPORTED_MODEL_FORMAT, target, access, reason="unknown_dtype", dtype="F8_E8M0"
    )


def test_malformed_header(model_dir: ModelDir) -> None:
    target, access = model_dir(tensors=None)
    with (target / "model.safetensors").open("wb") as handle:
        handle.write((1 << 20).to_bytes(8, "little"))
        handle.write(b'{"w":')
    _fails(ErrorCode.MODEL_METADATA_UNAVAILABLE, target, access)


def test_missing_and_invalid_config(model_dir: ModelDir) -> None:
    _fails(ErrorCode.MODEL_METADATA_UNAVAILABLE, *model_dir(config=None), reason="config_missing")
    target, access = model_dir(config=None, name="bad")
    (target / "config.json").write_bytes(b"{not json")
    _fails(ErrorCode.MODEL_METADATA_UNAVAILABLE, target, access, reason="invalid_config_json")


def _sharded(
    model_dir: ModelDir, st_writer: Any, weight_map: dict[str, str]
) -> tuple[Path, SourceAccess]:
    target, access = model_dir(tensors=None)
    st_writer.write_safetensors(
        target / "model-00001-of-00001.safetensors",
        {"model.embed_tokens.weight": ("BF16", [16, 8]), "lm_head.weight": ("BF16", [16, 8])},
    )
    index = {"metadata": {"total_size": 512}, "weight_map": weight_map}
    (target / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    return target, access


def test_sharded_index(model_dir: ModelDir, st_writer: Any) -> None:
    shard = "model-00001-of-00001.safetensors"
    inv = _inspect(
        *_sharded(
            model_dir, st_writer, {"model.embed_tokens.weight": shard, "lm_head.weight": shard}
        )
    )
    assert inv.index_total_size == 512
    assert {t.shard for t in inv.tensors} == {shard}


def test_index_and_headers_must_agree(model_dir: ModelDir, st_writer: Any) -> None:
    shard = "model-00001-of-00001.safetensors"
    target, access = _sharded(
        model_dir,
        st_writer,
        {"model.embed_tokens.weight": shard, "lm_head.weight": shard, "extra.weight": shard},
    )
    _fails(ErrorCode.MODEL_METADATA_UNAVAILABLE, target, access, reason="index_header_mismatch")


@pytest.mark.parametrize("shard", ["model-00002-of-00002.safetensors", "../outside.safetensors"])
def test_index_pointing_outside_the_source(model_dir: ModelDir, st_writer: Any, shard: str) -> None:
    target, access = _sharded(model_dir, st_writer, {"model.embed_tokens.weight": shard})
    _fails(ErrorCode.MODEL_METADATA_UNAVAILABLE, target, access, reason="index_shard_missing")


def test_local_change_after_resolution_is_detected(model_dir: ModelDir) -> None:
    target, access = model_dir()
    source = resolve_model(ModelSourceRef(reference=f"local:models/{target.name}"), access)
    (target / "config.json").write_text(json.dumps({**CONFIG, "vocab_size": 17}), encoding="utf-8")
    with pytest.raises(EstimatorError) as exc:
        inspect_model(source, access)
    assert exc.value.issue.code is ErrorCode.SOURCE_REVISION_CHANGED


def test_not_a_model_source(model_dir: ModelDir) -> None:
    from vramforge_estimator.schemas import DatasetSourceRef
    from vramforge_estimator.sources import resolve_dataset

    target, access = model_dir()
    source = resolve_dataset(DatasetSourceRef(reference=f"local:models/{target.name}"), access)
    with pytest.raises(EstimatorError) as exc:
        inspect_model(source, access)
    assert exc.value.issue.code is ErrorCode.INVALID_REQUEST


# ---------------------------------------------------------------- HF path through the fake hub


def test_hf_download_integrity_is_checked(mimo_hub: tuple[Any, SourceAccess, Any]) -> None:
    source, access, client = mimo_hub
    config = json.loads(client.repos[("model", source.repo_id)].files["config.json"])
    config["text_config"]["num_hidden_layers"] = 30
    client.tamper["config.json"] = json.dumps(config).encode()
    with pytest.raises(EstimatorError) as exc:
        inspect_model(source, access)
    assert exc.value.issue.code is ErrorCode.MODEL_METADATA_UNAVAILABLE
    assert exc.value.issue.details["reason"] == "integrity_mismatch"


def test_hf_errors_during_inspection(mimo_hub: tuple[Any, SourceAccess, Any]) -> None:
    source, access, client = mimo_hub
    response = httpx.Response(401, request=httpx.Request("GET", "https://huggingface.co/x"))
    client.errors["download_file"] = hf_errors.GatedRepoError("gated", response=response)
    with pytest.raises(EstimatorError) as exc:
        inspect_model(source, access)
    assert exc.value.issue.code is ErrorCode.SOURCE_ACCESS_DENIED
    assert exc.value.issue.stage is not None and exc.value.issue.stage.value == "inspecting"

    client.errors = {"read_safetensors_header": httpx.ReadTimeout("slow")}
    with pytest.raises(EstimatorError) as exc:
        inspect_model(source, access)
    assert exc.value.issue.code is ErrorCode.MODEL_METADATA_UNAVAILABLE
    assert exc.value.issue.retryable is True


def test_hf_header_with_unknown_dtype(mimo_hub: tuple[Any, SourceAccess, Any]) -> None:
    source, access, client = mimo_hub
    repo = client.repos[("model", source.repo_id)]
    shard = "model-00001-of-00004.safetensors"
    header, size, sha = repo.shards[shard]
    broken = json.loads(json.dumps(header))
    broken["lm_head.weight"]["dtype"] = "U32"
    repo.shards[shard] = (broken, size, sha)
    with pytest.raises(EstimatorError) as exc:
        inspect_model(source, access)
    assert exc.value.issue.code is ErrorCode.UNSUPPORTED_MODEL_FORMAT


def test_hub_client_is_not_used_for_local_sources(
    model_dir: ModelDir, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(access: SourceAccess) -> None:
        raise AssertionError("local sources must not touch the network")

    monkeypatch.setattr(hub, "get_hub_client", boom)
    _inspect(*model_dir())


def test_single_file_wins_over_index_like_transformers(model_dir: ModelDir, st_writer: Any) -> None:
    shard = "model-00001-of-00001.safetensors"
    target, access = _sharded(model_dir, st_writer, {"model.embed_tokens.weight": shard})
    st_writer.write_safetensors(target / "model.safetensors", TENSORS)
    inv = _inspect(target, access)
    assert inv.facts.extra["weights_file"] == "model.safetensors"
    assert {t.shard for t in inv.tensors} == {"model.safetensors"}
    assert inv.index_total_size is None


def test_explicit_transformers_weights(model_dir: ModelDir, st_writer: Any) -> None:
    config = {**CONFIG, "transformers_weights": "weights/custom.safetensors"}
    target, access = model_dir(config=config, tensors=None)
    st_writer.write_safetensors(target / "weights" / "custom.safetensors", TENSORS)
    inv = _inspect(target, access)
    assert inv.facts.extra["weights_file"] == "weights/custom.safetensors"
    bad_values = ("pytorch_model.bin", "../outside.safetensors", "missing.safetensors", 3)
    for i, bad in enumerate(bad_values):
        target, access = model_dir(
            {"pytorch_model.bin": b"\x80"},
            config={**CONFIG, "transformers_weights": bad},
            name=f"bad{i}",
        )
        _fails(
            ErrorCode.UNSUPPORTED_MODEL_FORMAT,
            target,
            access,
            reason="explicit_weights_unsupported",
        )
