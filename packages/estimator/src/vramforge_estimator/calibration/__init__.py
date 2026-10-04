"""Calibration registry (plan.md §11.4, §17.4).

No GPU measurements exist yet (M5), so every lookup is a miss and estimates stay `analytic`.
Callers must not upgrade evidence to `calibrated` without a registry hit.
"""

from __future__ import annotations

from dataclasses import dataclass

CALIBRATION_VERSION = "none-0"


@dataclass(frozen=True)
class CalibrationHit:
    profile_key: str
    correction_bytes: int
    domain_note: str


def lookup(profile_key: str, *, max_sequence_length: int, microbatch: int) -> CalibrationHit | None:
    """Return a calibration for this profile/shape domain, or None (always None for now)."""
    del profile_key, max_sequence_length, microbatch
    return None


__all__ = ["CALIBRATION_VERSION", "CalibrationHit", "lookup"]
