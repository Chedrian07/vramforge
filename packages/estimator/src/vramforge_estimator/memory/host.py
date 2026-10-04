"""Implementation module (stub until implemented by the owning agent)."""

from __future__ import annotations

from vramforge_estimator.schemas import (
    AnalysisRamEstimate,
    DiskEstimate,
    HostRamEstimate,
    ModelInventory,
    ResolvedConfig,
    SourceManifest,
)


def estimate_host_ram(
    inventory: ModelInventory, cfg: ResolvedConfig, model_source: SourceManifest | None
) -> HostRamEstimate:
    raise NotImplementedError


def estimate_disk(
    inventory: ModelInventory,
    cfg: ResolvedConfig,
    model_source: SourceManifest | None,
    dataset_source: SourceManifest | None,
    artifact_bytes: int | None,
) -> DiskEstimate:
    raise NotImplementedError


def estimate_analysis_ram(rows: int | None, tokenizer_bytes: int | None) -> AnalysisRamEstimate:
    raise NotImplementedError
