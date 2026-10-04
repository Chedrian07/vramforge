"""Objective-specific preprocessing that mirrors the pinned TRL version.

Owner: scan agent (docs/architecture.md §8). Signatures are the contract.
"""

from __future__ import annotations

from .base import PreprocessingAdapter, TokenizedRecord
from .registry import get_adapter

__all__ = ["PreprocessingAdapter", "TokenizedRecord", "get_adapter"]
