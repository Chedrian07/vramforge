"""Inspection contracts: tokenizer handle and dataset row stream (plan.md §6.2, §7.1, §7.6)."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from vramforge_estimator.schemas import TokenizerManifest


@dataclass(frozen=True)
class TokenizerHandle:
    """A loaded tokenizer (transformers PreTrainedTokenizerBase, no torch) plus its manifest.

    `tokenizer` is typed `Any` so the core does not import transformers at module import time.
    Remote code is never trusted (trust_remote_code=False).
    """

    tokenizer: Any
    manifest: TokenizerManifest


@dataclass(frozen=True)
class SourceRow:
    row_index: int  # 0-based index in source order within the split
    shard_id: str  # file/shard the row came from
    row: Mapping[str, Any]


class RowStream(Protocol):
    """Sequential reader over one split of a finite snapshot.

    Iterating yields every row of every shard in a deterministic order. `complete` becomes True
    only after every shard reached EOF; quota stops, parse failures and cancellation leave it False
    (plan §7.7).
    """

    split: str
    config: str | None
    shards: list[str]
    total_rows: int | None  # None when unknown in advance

    def __iter__(self) -> Iterator[SourceRow]: ...

    @property
    def complete(self) -> bool: ...

    @property
    def shards_completed(self) -> int: ...
