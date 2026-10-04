"""Implementation module (stub until implemented by the owning agent)."""

from __future__ import annotations

from vramforge_estimator.schemas import ArchitectureFacts

from .base import ArchitectureAdapter


def match_adapter(facts: ArchitectureFacts) -> str | None:
    """Return the adapter id that supports these facts, or None (UNSUPPORTED_ARCHITECTURE).
    Matching is structural (config fields/layer types), never by model-name substrings."""
    raise NotImplementedError


def get_adapter(adapter_id: str) -> ArchitectureAdapter:
    raise NotImplementedError
