"""Backend profile registry and compatibility resolution (plan.md §11).

Profiles live in the repository `profiles/` directory (analytic, environments, hardware) and are
loaded from `VRAMFORGE_PROFILES_DIR` (default: repo/profiles, /app/profiles in containers).

Owner: memory agent (docs/architecture.md §8). Signatures are the contract.
"""

from __future__ import annotations

from .resolver import backend_profiles, resolve, support_for, validate_request

__all__ = ["backend_profiles", "resolve", "support_for", "validate_request"]
