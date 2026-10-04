"""ModelInspector: config + safetensors headers → `ModelInventory` (plan.md §6.2–6.5).

Order: weight-format triage (only transformers-loadable safetensors; GGUF, pickle ``*.bin``,
adapter-only and pre-quantized checkpoints are refused, pickles are never opened) → config.json
→ remote-code check → index → every shard header (no weights) → inventory.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import PurePosixPath
from typing import Any

from vramforge_estimator.errors import EstimatorError, make_issue
from vramforge_estimator.schemas import ErrorCode, ModelInventory, Stage
from vramforge_estimator.sources import ResolvedSource, SourceAccess

from .model_config import load_config, quantization_method, requires_remote_code
from .model_files import SourceFiles, header_error, open_source_files
from .model_inventory import ShardTensors, build_inventory
from .safetensors_header import HeaderError, validate_header

INDEX_FILE = "model.safetensors.index.json"
SINGLE_FILE = "model.safetensors"
MAX_PARALLEL_HEADERS = 8

_FORMAT_MESSAGES = {
    "adapter_only": "LoRA/PEFT adapter만 들어 있는 저장소입니다. base 모델을 입력하세요.",
    "gguf": "GGUF checkpoint는 지원하지 않습니다. Transformers용 safetensors 가중치가 필요합니다. "
    "(사전 양자화 파일은 bitsandbytes QLoRA 입력으로 인정하지 않습니다.)",
    "pytorch_pickle": "pickle 기반 *.bin 가중치만 있는 checkpoint는 지원하지 않습니다. "
    "보안상 pickle 파일을 열지 않으므로 safetensors 가중치가 필요합니다.",
    "pickle": "pickle 기반 가중치(*.pt, *.pth, *.ckpt)만 있는 checkpoint는 지원하지 않습니다.",
    "tensorflow": "TensorFlow(*.h5) 가중치만 있는 checkpoint는 지원하지 않습니다.",
    "flax": "Flax(*.msgpack) 가중치만 있는 checkpoint는 지원하지 않습니다.",
    "onnx": "ONNX 모델은 지원하지 않습니다.",
    "nonstandard_safetensors": "Transformers가 읽는 이름(model.safetensors 또는 "
    "model.safetensors.index.json)의 safetensors 가중치가 없습니다.",
    "no_weights": "가중치 파일을 찾을 수 없습니다.",
}


def _error(code: ErrorCode, message: str, **details: object) -> EstimatorError:
    return EstimatorError(
        make_issue(code, message, stage=Stage.INSPECTING, component="model", **details)
    )


def other_weight_formats(paths: set[str]) -> list[str]:
    """Non-safetensors weight formats present in the file list (sorted)."""
    found: set[str] = set()
    for path in paths:
        name = PurePosixPath(path).name
        suffix = PurePosixPath(path).suffix.lower()
        if suffix == ".gguf":
            found.add("gguf")
        elif suffix == ".bin" and name not in ("training_args.bin",):
            found.add("pytorch_pickle")
        elif suffix in (".pt", ".pth", ".ckpt", ".pkl", ".pickle"):
            found.add("pickle")
        elif suffix == ".h5":
            found.add("tensorflow")
        elif suffix == ".msgpack":
            found.add("flax")
        elif suffix == ".onnx":
            found.add("onnx")
    return sorted(found)


def triage_weights(paths: set[str]) -> str:
    """ "index" or "single", or raise UNSUPPORTED_MODEL_FORMAT (plan §6.3, §18)."""
    if INDEX_FILE in paths:
        return "index"
    if SINGLE_FILE in paths:
        return "single"
    others = other_weight_formats(paths)
    if "adapter_config.json" in paths:
        fmt = "adapter_only"
    elif others:
        fmt = next(
            f
            for f in ("gguf", "pytorch_pickle", "pickle", "tensorflow", "flax", "onnx")
            if f in others
        )
    elif any(p.endswith(".safetensors") for p in paths):
        fmt = "nonstandard_safetensors"
    else:
        fmt = "no_weights"
    raise _error(
        ErrorCode.UNSUPPORTED_MODEL_FORMAT,
        _FORMAT_MESSAGES[fmt],
        reason="unsupported_weight_format",
        format=fmt,
    )


def _load_json(files: SourceFiles, path: str) -> dict[str, Any]:
    try:
        value = json.loads(files.read(path).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        value = None
    if not isinstance(value, dict):
        raise _error(
            ErrorCode.MODEL_METADATA_UNAVAILABLE,
            f"{path}을(를) JSON 객체로 읽을 수 없습니다.",
            reason="invalid_json",
            file=path,
        )
    return value


def _index_shards(index: dict[str, Any], paths: set[str]) -> tuple[dict[str, str], int | None]:
    weight_map = index.get("weight_map")
    valid = (
        isinstance(weight_map, dict)
        and bool(weight_map)
        and all(isinstance(k, str) and isinstance(v, str) for k, v in weight_map.items())
    )
    if not valid:
        raise _error(
            ErrorCode.MODEL_METADATA_UNAVAILABLE,
            "safetensors index의 weight_map 형식이 올바르지 않습니다.",
            reason="invalid_index",
        )
    for shard in set(weight_map.values()):
        parts = PurePosixPath(shard).parts
        if shard.startswith("/") or ".." in parts or shard not in paths:
            raise _error(
                ErrorCode.MODEL_METADATA_UNAVAILABLE,
                "safetensors index가 source에 없는 shard 파일을 가리킵니다.",
                reason="index_shard_missing",
            )
    metadata = index.get("metadata")
    total = metadata.get("total_size") if isinstance(metadata, dict) else None
    total = total if isinstance(total, int) and not isinstance(total, bool) else None
    return weight_map, total


def _read_shards(files: SourceFiles, shards: list[str]) -> list[ShardTensors]:
    def read(shard: str) -> ShardTensors:
        header, payload_size = files.header(shard)
        try:
            tensors, _ = validate_header(header, payload_size=payload_size)
        except HeaderError as exc:
            raise header_error(shard, exc) from None
        return ShardTensors(shard=shard, tensors=tensors)

    if len(shards) == 1:
        return [read(shards[0])]
    with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_HEADERS, len(shards))) as pool:
        return list(pool.map(read, shards))


def inspect_model(source: ResolvedSource, access: SourceAccess) -> ModelInventory:
    """Build the tensor inventory from config + safetensors headers (no weight download).

    Raises `EstimatorError` with MODEL_METADATA_UNAVAILABLE, REMOTE_CODE_REQUIRED or
    UNSUPPORTED_MODEL_FORMAT.
    """
    if source.kind != "model":
        raise _error(ErrorCode.INVALID_REQUEST, "모델 source가 아닙니다.", reason="not_a_model")
    files = open_source_files(source, access)
    paths = set(files.entries())
    layout = triage_weights(paths)
    if "config.json" not in paths:
        raise _error(
            ErrorCode.MODEL_METADATA_UNAVAILABLE,
            "config.json이 없어 모델 구조를 확인할 수 없습니다.",
            reason="config_missing",
        )
    config = load_config(files.read("config.json"))
    if requires_remote_code(config):
        raise _error(
            ErrorCode.REMOTE_CODE_REQUIRED,
            "이 모델은 저장소에 포함된 사용자 코드(trust_remote_code)가 있어야 로드됩니다. "
            "원격 코드는 실행하지 않으므로 지원하지 않습니다.",
            reason="auto_map_without_native_class",
            model_type=config.model_type,
        )
    method = quantization_method(config)
    if method is not None:
        raise _error(
            ErrorCode.UNSUPPORTED_MODEL_FORMAT,
            f"이미 양자화된 checkpoint({method})는 지원하지 않습니다. "
            "원본 정밀도(bf16/fp16/fp32) safetensors 가중치가 필요합니다.",
            reason="prequantized_checkpoint",
            quantized_checkpoint_format=method,
        )

    index_total_size: int | None = None
    weight_map: dict[str, str] | None = None
    if layout == "index":
        weight_map, index_total_size = _index_shards(_load_json(files, INDEX_FILE), paths)
        shard_names = sorted(set(weight_map.values()))
    else:
        shard_names = [SINGLE_FILE]

    shards = _read_shards(files, shard_names)
    if weight_map is not None:
        located = {t.name: s.shard for s in shards for t in s.tensors}
        if located != weight_map:
            raise _error(
                ErrorCode.MODEL_METADATA_UNAVAILABLE,
                "safetensors index와 shard header의 tensor 목록이 일치하지 않습니다.",
                reason="index_header_mismatch",
                index_tensors=len(weight_map),
                header_tensors=len(located),
            )

    others = other_weight_formats(paths)
    return build_inventory(
        config,
        shards,
        index_total_size=index_total_size,
        extra_facts={"ignored_weight_formats": others} if others else None,
    )
