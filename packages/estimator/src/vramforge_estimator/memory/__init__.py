"""Memory engine: timepoint evaluation, scenario estimates, margin, hardware fit, RAM/disk.

Owner: memory agent (docs/architecture.md §8). Signatures are the contract.
"""

from __future__ import annotations

from .engine import assess_fit, evaluate, recommend
from .estimate import estimate_memory
from .host import estimate_analysis_ram, estimate_disk, estimate_host_ram

__all__ = [
    "assess_fit",
    "estimate_analysis_ram",
    "estimate_disk",
    "estimate_host_ram",
    "estimate_memory",
    "evaluate",
    "recommend",
]
