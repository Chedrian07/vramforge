"""Source manifests, tokenizer manifest and model inventory (plan.md §6, §16.2).

A manifest pins a source to an immutable identity (HF commit sha, or a content digest for local
files/uploads). Display fields must never contain host absolute paths or credentials.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import Field

from .common import SourceType, VFModel


class FileEntry(VFModel):
    path: str  # repo-relative / root-relative path
    size: int | None = None
    sha256: str | None = None
    blob_id: str | None = None  # git blob / LFS oid when known


class SourceManifest(VFModel):
    kind: Literal["model", "dataset"]
    source_type: SourceType
    reference: str  # normalized, display-safe ("hf:org/name", "local:<root>/<rel>", "upload:<id>")
    repo_id: str | None = None
    requested_revision: str | None = None
    resolved_revision: str  # commit sha (HF) or "sha256:<content-manifest-digest>"
    files: list[FileEntry] = Field(default_factory=list)
    private: bool | None = None
    gated: bool | None = None
    last_modified: str | None = None
    fingerprint: str
    notes: list[str] = Field(default_factory=list)


class TokenizerManifest(VFModel):
    tokenizer_class: str
    vocab_size: int  # len(tokenizer), incl. added tokens
    config_vocab_size: int | None = None  # embedding rows declared by the model config
    bos_token: str | None = None
    eos_token: str | None = None
    pad_token: str | None = None
    adds_bos_by_default: bool | None = None
    adds_eos_by_default: bool | None = None
    model_max_length: int | None = None
    model_max_length_is_sentinel: bool = False
    chat_template_present: bool
    chat_template_source: Literal[
        "chat_template.jinja", "tokenizer_config.json", "processor", "none"
    ] = "none"
    chat_template_sha256: str | None = None
    has_generation_markers: bool = False  # {% generation %} blocks (assistant-only loss)
    template_kwargs: list[str] = Field(default_factory=list)  # e.g. ["enable_thinking"]
    template_parse_error: str | None = None  # Jinja parse failure (display-safe), if any
    files_sha256: dict[str, str] = Field(default_factory=dict)
    fingerprint: str


class ModelComponent(StrEnum):
    TEXT = "text"
    VISION = "vision"
    AUDIO = "audio"
    PROJECTOR = "projector"
    MTP = "mtp"
    OTHER = "other"


class TensorRole(StrEnum):
    EMBEDDING = "embedding"
    LM_HEAD = "lm_head"
    LINEAR_WEIGHT = "linear_weight"
    LINEAR_BIAS = "linear_bias"
    NORM = "norm"
    CONV = "conv"
    PARAMETER = "parameter"  # e.g. A_log, dt_bias, position embeddings
    OTHER = "other"


class TensorInfo(VFModel):
    name: str
    dtype: str  # canonical dtype (units.canonical_dtype)
    shape: list[int]
    numel: int
    nbytes: int
    shard: str | None = None
    component: ModelComponent
    role: TensorRole
    module: str  # parent module path, e.g. "model.language_model.layers.3.self_attn.q_proj"
    layer_index: int | None = None


class LinearModule(VFModel):
    """A 2-D Linear-like module: the unit for quantization and LoRA accounting."""

    name: str  # module path without ".weight"
    kind: str  # leaf name, e.g. "q_proj", "in_proj_qkv", "down_proj", "qkv"
    in_features: int
    out_features: int
    has_bias: bool
    component: ModelComponent
    layer_index: int | None = None
    layer_type: str | None = None  # "full_attention" | "linear_attention" | "mlp" | "vision" | ...
    dtype: str


class LayerTypeCount(VFModel):
    layer_type: str
    count: int


class ArchitectureFacts(VFModel):
    architectures: list[str]
    model_type: str
    text_model_type: str | None = None
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int | None = None
    vocab_size: int
    num_attention_heads: int | None = None
    num_key_value_heads: int | None = None
    head_dim: int | None = None
    head_dim_source: Literal["explicit", "inferred", "unknown"] = "unknown"
    layer_types: list[str] = Field(default_factory=list)  # one entry per text decoder layer
    layer_type_counts: list[LayerTypeCount] = Field(default_factory=list)
    sliding_window: int | None = None
    # Linear-attention dims (e.g. Gated DeltaNet): num_key_heads, num_value_heads,
    # key_head_dim, value_head_dim, conv_kernel_dim.
    linear_attention: dict[str, int] = Field(default_factory=dict)
    max_position_embeddings: int | None = None
    tie_word_embeddings: bool = False
    config_dtype: str | None = None
    has_vision: bool = False
    has_mtp: bool = False
    rope: dict[str, Any] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)  # other config fields adapters may need
    config_sha256: str


class ComponentParams(VFModel):
    component: ModelComponent
    params: int
    bytes_serialized: int


class ModelInventory(VFModel):
    """Full inventory from safetensors headers + config (stored as an artifact)."""

    facts: ArchitectureFacts
    tensors: list[TensorInfo]
    linear_modules: list[LinearModule]
    params_total: int  # unique logical parameters (tied tensors counted once)
    bytes_serialized_total: int
    by_component: list[ComponentParams]
    tied_groups: list[list[str]] = Field(default_factory=list)
    # Pre-quantized checkpoints (gptq/awq/gguf/fp8/bnb...) are not accepted as bnb QLoRA input.
    quantized_checkpoint_format: str | None = None
    index_total_size: int | None = None  # informational only (plan §6.2)
    inventory_hash: str


class ModelInventorySummary(VFModel):
    """The part of the inventory embedded in results and API responses."""

    facts: ArchitectureFacts
    tensor_count: int
    linear_module_count: int
    params_total: int
    bytes_serialized_total: int
    by_component: list[ComponentParams]
    tied_groups: list[list[str]] = Field(default_factory=list)
    quantized_checkpoint_format: str | None = None
    inventory_hash: str
