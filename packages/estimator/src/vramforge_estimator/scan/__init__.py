"""Full dataset scan, row-length artifact, statistics, preservation audit, context validation.

Owner: scan agent (docs/architecture.md §8). Signatures are the contract.
"""

from __future__ import annotations

from .base import LengthTable, ScanContext, ScanLimits, ScanOutcome
from .scanner import full_scan, load_lengths
from .validation import audit_preservation, validate_context

__all__ = [
    "LengthTable",
    "ScanContext",
    "ScanLimits",
    "ScanOutcome",
    "audit_preservation",
    "full_scan",
    "load_lengths",
    "validate_context",
]
