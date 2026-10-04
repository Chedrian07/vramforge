"""Backend profile registry (plan.md §11.2): environment, analytic and hardware profiles.

Profiles are versioned YAML files. The directory is `VRAMFORGE_PROFILES_DIR` when set, otherwise the
repository `profiles/` directory next to the package sources (source checkout), otherwise
`/app/profiles` (container image). Every file is validated on load; a profile that does not
validate is a deployment error, never silently skipped.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from vramforge_estimator.schemas import (
    EvidenceLevel,
    GpuPreset,
    Objective,
    OptimizerName,
    ReferenceStrategy,
    RolloutBackend,
    Strategy,
    TrainingReadiness,
)
from vramforge_estimator.units import GiB

PROFILES_DIR_ENV = "VRAMFORGE_PROFILES_DIR"
CONTAINER_PROFILES_DIR = Path("/app/profiles")


def _repo_profiles_dir() -> Path | None:
    # <repo>/packages/estimator/src/vramforge_estimator/compatibility/profiles.py -> <repo>/profiles
    parents = Path(__file__).resolve().parents
    return parents[5] / "profiles" if len(parents) > 5 else None


class ProfileError(ValueError):
    """A profile file is missing or invalid (deployment error)."""


# ---------------------------------------------------------------- dependency lock digest


def normalize_package_name(name: str) -> str:
    """PEP 503 normalized distribution name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def lock_digest(packages: Mapping[str, str]) -> str:
    """sha256 over the sorted, newline-terminated `name==version` lines (PEP 503 names)."""
    lines = sorted(f"{normalize_package_name(name)}=={ver}" for name, ver in packages.items())
    payload = "".join(f"{line}\n" for line in lines)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- models


class _Profile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class KernelInventory(_Profile):
    """Optional kernel packages of the training environment (they select runtime code paths)."""

    installed: list[str] = Field(default_factory=list)
    absent: list[str] = Field(default_factory=list)
    linear_attention: Literal["torch_fallback", "fla"]
    logprob_kernel: Literal["trl_triton_fused", "torch_fallback"]
    notes: list[str] = Field(default_factory=list)


class TrainerContract(_Profile):
    """Config fields the exporter must pin for strict no-truncation, and fields removed in TRL."""

    config_class: str
    no_truncation: dict[str, Any] = Field(default_factory=dict)
    removed_fields: list[str] = Field(default_factory=list)
    forbidden_template_kwargs: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class EnvironmentProfile(_Profile):
    id: str
    version: str
    description: str
    platform: str
    python: str
    cuda: str | None = None
    packages: dict[str, str]
    kernels: KernelInventory
    dependency_lock_digest: str
    trainer_defaults: dict[str, Any] = Field(default_factory=dict)
    trainers: dict[Objective, TrainerContract]
    sources: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_digest(self) -> EnvironmentProfile:
        expected = lock_digest(self.packages)
        if self.dependency_lock_digest != expected:
            raise ValueError(
                f"dependency_lock_digest {self.dependency_lock_digest} does not match the "
                f"packages ({expected})"
            )
        return self


class SupportRule(_Profile):
    objective: Objective
    strategy: Strategy
    grade: EvidenceLevel | None  # None = unsupported
    readiness: TrainingReadiness
    note: str = ""

    @model_validator(mode="after")
    def _unsupported_needs_reason(self) -> SupportRule:
        if self.grade is None and (
            self.readiness is not TrainingReadiness.UNSUPPORTED or not self.note
        ):
            raise ValueError("unsupported entries need readiness=unsupported and a reason")
        return self


class ScopeRule(_Profile):
    architecture: str  # config.architectures[0]
    scope: Literal["full_checkpoint", "text_only"]
    reason: str


class TextOnlyRule(_Profile):
    verified: bool
    reason: str


class LoadingRules(_Profile):
    entrypoint: Literal["trl_model_id"]
    architecture_source: str
    scope_rules: list[ScopeRule] = Field(default_factory=list)
    default_scope: Literal["full_checkpoint"]
    default_scope_reason: str
    text_only: TextOnlyRule
    default_load_dtype: Literal["bfloat16", "float32"]
    load_dtypes: dict[str, str]  # supported load dtype -> note
    device_map: Literal["auto"]
    device_map_budget_factor: dict[Literal["quantized", "dense"], float]
    device_map_note: str = ""


class QuantizationPreset(_Profile):
    library: Literal["bitsandbytes"]
    formats: list[Literal["nf4", "fp4"]]
    default_format: Literal["nf4", "fp4"]
    double_quant: bool
    compute_dtype: str
    supported_compute_dtypes: list[str]
    quant_storage: str
    blocksize: int
    nested_blocksize: int
    skip_modules: list[str]
    note: str = ""


class AttentionRules(_Profile):
    by_layer_type: dict[str, str]  # default path per layer type
    supported: dict[str, list[str]]


class LossRule(_Profile):
    default: str
    supported: list[str]
    note: str = ""


class CheckpointingRules(_Profile):
    default_enabled: bool
    granularity: Literal["per_decoder_layer"]
    use_reentrant: bool
    note: str = ""


class OptimizerRule(_Profile):
    name: OptimizerName
    states_per_param: int
    state_dtype: Literal["param", "uint8"]
    eight_bit: bool = False
    block_size: int | None = None
    min_8bit_size: int | None = None
    paged: bool = False
    fused: bool = False
    note: str = ""


class BatchPreset(_Profile):
    microbatch: int = Field(ge=1)
    accumulation: int = Field(ge=1)


class UnsupportedOption(_Profile):
    option: str
    reason: str


class WorkspaceRange(_Profile):
    low: float = Field(ge=0)
    high: float = Field(ge=0)
    source: str
    note: str = ""

    @model_validator(mode="after")
    def _ordered(self) -> WorkspaceRange:
        if self.low > self.high:
            raise ValueError("workspace range low must not exceed high")
        return self


class WorkspaceRules(_Profile):
    cuda_context_bytes: WorkspaceRange
    library_workspace_bytes: WorkspaceRange
    allocator_slack_fraction: WorkspaceRange


class AnalyticProfile(_Profile):
    id: str
    version: str
    description: str
    architecture_adapter: str
    environment: str
    model_types: list[str]
    architectures: list[str]
    trainers: dict[Objective, str]
    support: list[SupportRule]
    loading: LoadingRules
    quantization: QuantizationPreset
    attention: AttentionRules
    loss: dict[Objective, LossRule]
    checkpointing: CheckpointingRules
    optimizers: list[OptimizerRule]
    dpo_reference_auto: dict[Literal["peft", "full"], ReferenceStrategy]
    rollout_backends: list[RolloutBackend]
    presets: dict[Objective, BatchPreset]
    unsupported_options: list[UnsupportedOption] = Field(default_factory=list)
    workspace: WorkspaceRules
    calibration_coverage: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _complete(self) -> AnalyticProfile:
        keys = [(s.objective, s.strategy) for s in self.support]
        expected = {(o, s) for o in Objective for s in Strategy}
        if len(keys) != len(set(keys)) or set(keys) != expected:
            raise ValueError("support must list every objective x strategy exactly once")
        for mapping, what in (
            (self.trainers, "trainers"),
            (self.loss, "loss"),
            (self.presets, "presets"),
        ):
            if set(mapping) != set(Objective):
                raise ValueError(f"{what} must cover every objective")
        return self

    def support_rule(self, objective: Objective, strategy: Strategy) -> SupportRule:
        return next(s for s in self.support if s.objective is objective and s.strategy is strategy)

    def optimizer_rule(self, name: OptimizerName) -> OptimizerRule | None:
        return next((o for o in self.optimizers if o.name is name), None)


class GpuProfile(_Profile):
    id: str
    name: str
    nominal_gib: int = Field(ge=1)
    note: str = ""

    @property
    def total_bytes(self) -> int:
        return self.nominal_gib * GiB


class HardwareCatalog(_Profile):
    version: str
    note: str
    gpus: list[GpuProfile]


# ---------------------------------------------------------------- registry


@dataclass(frozen=True)
class ProfileRegistry:
    root: Path
    environments: Mapping[str, EnvironmentProfile]
    analytic: Mapping[str, AnalyticProfile]
    hardware: HardwareCatalog

    def environment_for(self, profile: AnalyticProfile) -> EnvironmentProfile:
        return self.environments[profile.environment]

    def profile_for_adapter(self, adapter_id: str) -> AnalyticProfile | None:
        return next(
            (p for p in self.analytic.values() if p.architecture_adapter == adapter_id), None
        )

    def gpu(self, preset_id: str) -> GpuProfile | None:
        return next((g for g in self.hardware.gpus if g.id == preset_id), None)

    def gpu_presets(self) -> list[GpuPreset]:
        note = self.hardware.note
        return [
            GpuPreset(
                id=g.id,
                name=g.name,
                total_bytes=g.total_bytes,
                note=f"{g.note} {note}".strip() if g.note else note,
            )
            for g in self.hardware.gpus
        ]


def default_profiles_dir() -> Path:
    configured = os.environ.get(PROFILES_DIR_ENV)
    if configured:
        return Path(configured)
    repo = _repo_profiles_dir()
    if repo is not None and repo.is_dir():
        return repo
    return CONTAINER_PROFILES_DIR


def load_registry(root: Path | None = None) -> ProfileRegistry:
    directory = (root if root is not None else default_profiles_dir()).resolve()
    return _load_cached(str(directory))


def clear_registry_cache() -> None:
    _load_cached.cache_clear()


def _read_yaml(path: Path, base: Path) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ProfileError(f"cannot read profile {path.relative_to(base)}: {exc}") from exc


def _parse[M: BaseModel](model: type[M], path: Path, base: Path) -> M:
    try:
        return model.model_validate(_read_yaml(path, base))
    except ValidationError as exc:
        raise ProfileError(f"invalid profile {path.relative_to(base)}: {exc}") from exc


@lru_cache(maxsize=8)
def _load_cached(directory: str) -> ProfileRegistry:
    base = Path(directory)
    if not base.is_dir():
        raise ProfileError(f"profiles directory not found (set {PROFILES_DIR_ENV})")
    environments: dict[str, EnvironmentProfile] = {}
    for path in sorted((base / "environments").glob("*.yaml")):
        env = _parse(EnvironmentProfile, path, base)
        if env.id in environments:
            raise ProfileError(f"duplicate environment id {env.id}")
        environments[env.id] = env
    analytic: dict[str, AnalyticProfile] = {}
    for path in sorted((base / "analytic").glob("*.yaml")):
        prof = _parse(AnalyticProfile, path, base)
        if prof.id in analytic:
            raise ProfileError(f"duplicate profile id {prof.id}")
        if prof.environment not in environments:
            raise ProfileError(
                f"profile {prof.id} references unknown environment {prof.environment}"
            )
        analytic[prof.id] = prof
    adapters = [p.architecture_adapter for p in analytic.values()]
    if len(adapters) != len(set(adapters)):
        raise ProfileError("each architecture adapter must have exactly one analytic profile")
    hardware_path = base / "hardware" / "gpus.yaml"
    if not hardware_path.is_file():
        raise ProfileError("hardware/gpus.yaml is missing")
    hardware = _parse(HardwareCatalog, hardware_path, base)
    ids = [g.id for g in hardware.gpus]
    if len(ids) != len(set(ids)):
        raise ProfileError("duplicate GPU preset id")
    return ProfileRegistry(
        root=base, environments=environments, analytic=analytic, hardware=hardware
    )


__all__ = [
    "PROFILES_DIR_ENV",
    "AnalyticProfile",
    "EnvironmentProfile",
    "GpuProfile",
    "HardwareCatalog",
    "OptimizerRule",
    "ProfileError",
    "ProfileRegistry",
    "SupportRule",
    "WorkspaceRules",
    "clear_registry_cache",
    "default_profiles_dir",
    "load_registry",
    "lock_digest",
    "normalize_package_name",
]
