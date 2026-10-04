"""Layered cache keys (plan.md §16.3) and request fingerprints.

source_key ⊂ preprocess_key ⊂ batch_key ⊂ estimate_key. A change in a lower layer invalidates
every layer above it; a change that only affects an upper layer (e.g. GRPO completion budget)
keeps the tokenization cache.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel


def _canonical(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _canonical(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, list | tuple):
        return [_canonical(v) for v in value]
    return value


def stable_hash(*parts: Any, prefix: str = "") -> str:
    """sha256 over a canonical JSON encoding of `parts` (dict order independent)."""
    payload = json.dumps(_canonical(list(parts)), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{prefix}{digest}" if prefix else digest


def source_key(source_type: str, immutable_identity: str) -> str:
    return stable_hash(source_type, immutable_identity, prefix="src_")


def preprocess_key(
    *,
    source_key: str,
    config: str | None,
    split: str | None,
    mapping: Mapping[str, Any] | BaseModel | None,
    objective: str,
    tokenizer_fingerprint: str,
    chat_template_sha256: str | None,
    template_kwargs: Mapping[str, Any],
    adapter_name: str,
    adapter_version: str,
    dependency_lock_digest: str,
) -> str:
    return stable_hash(
        source_key,
        config,
        split,
        mapping,
        objective,
        tokenizer_fingerprint,
        chat_template_sha256,
        dict(template_kwargs),
        adapter_name,
        adapter_version,
        dependency_lock_digest,
        prefix="pre_",
    )


def batch_key(
    *,
    preprocess_key: str,
    collator: Mapping[str, Any],
    sampler: Mapping[str, Any],
    microbatch: int,
    pad_to_multiple_of: int | None,
    packing: bool,
    distribution: Mapping[str, Any],
) -> str:
    return stable_hash(
        preprocess_key,
        dict(collator),
        dict(sampler),
        microbatch,
        pad_to_multiple_of,
        packing,
        dict(distribution),
        prefix="bat_",
    )


def estimate_key(
    *,
    batch_key: str,
    inventory_hash: str,
    resolved: Mapping[str, Any] | BaseModel,
    margin_policy: Mapping[str, Any] | BaseModel,
    calibration_version: str,
) -> str:
    return stable_hash(
        batch_key, inventory_hash, resolved, margin_policy, calibration_version, prefix="est_"
    )


def request_fingerprint(request: BaseModel) -> str:
    """Fingerprint of a full request; the UI uses it to drop stale responses (plan §4.2)."""
    return stable_hash(request, prefix="req_")
