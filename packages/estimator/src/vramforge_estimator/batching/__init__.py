"""Batch planning (plan.md §8).

Owner: scan agent (docs/architecture.md §8). Signatures are the contract.
"""

from __future__ import annotations

from .planner import plan_batches

__all__ = ["plan_batches"]
