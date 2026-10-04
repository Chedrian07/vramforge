"""Batch planning (plan.md §8).

Owner: scan agent (docs/architecture.md §8). Signatures are the contract.
"""

from __future__ import annotations

from vramforge_estimator.scan import LengthTable
from vramforge_estimator.schemas import BatchPlan, ResolvedConfig


def plan_batches(lengths: LengthTable, resolved: ResolvedConfig, *, seed: int) -> BatchPlan:
    """Build the worst-case shape, the seeded-sampler maximum and (GRPO) one shape per completion
    budget. Reports sampler coverage (rows dropped by batch/group divisibility) as an issue."""
    raise NotImplementedError


__all__ = ["plan_batches"]
