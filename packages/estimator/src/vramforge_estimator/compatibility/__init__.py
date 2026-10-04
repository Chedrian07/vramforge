"""Backend profile registry and compatibility resolution (plan.md §11).

Profiles live in the repository `profiles/` directory (analytic, environments, hardware) and are
loaded from `VRAMFORGE_PROFILES_DIR` (default: repo/profiles, /app/profiles in containers).

Owner: memory agent (docs/architecture.md §8). Signatures are the contract.
"""

from __future__ import annotations

from vramforge_estimator.schemas import (
    AnalysisRequest,
    ArchitectureFacts,
    BackendProfilesResponse,
    CompatibilityReport,
    Issue,
    ModelInventory,
    ResolvedConfig,
    SupportEntry,
    TokenizerManifest,
)


def validate_request(request: AnalysisRequest) -> list[Issue]:
    """Cross-field checks that need no network (e.g. 4-bit + full fine-tune, LoRA + 4-bit,
    multi-GPU without a topology adapter, sample scan claims). Errors block job creation."""
    raise NotImplementedError


def resolve(
    request: AnalysisRequest,
    inventory: ModelInventory,
    tokenizer: TokenizerManifest | None,
) -> tuple[ResolvedConfig | None, CompatibilityReport]:
    """Pick the profile (data preservation → objective preservation → compatibility → memory
    efficiency, plan §11.1), resolve every knob and record requested vs resolved."""
    raise NotImplementedError


def support_for(facts: ArchitectureFacts) -> tuple[str | None, list[SupportEntry]]:
    """Adapter id (or None) and the objective × strategy support grades for these facts."""
    raise NotImplementedError


def backend_profiles() -> BackendProfilesResponse:
    raise NotImplementedError


__all__ = ["backend_profiles", "resolve", "support_for", "validate_request"]
