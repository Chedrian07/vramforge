"""Exact length statistics kept as integer counts (plan.md §7.7).

Lengths are small integers with many repeats, so a ``length -> count`` map gives exact
count/min/max/mean/quantiles/histogram in memory proportional to the number of distinct lengths,
never to the number of rows. Quantiles use the nearest-rank definition (an observed value; equal
to numpy ``method="inverted_cdf"``, the ``nearest_rank`` block of the golden example stats).
"""

from __future__ import annotations

import heapq
from collections.abc import Callable, Iterator
from typing import Any

from vramforge_estimator.preprocessing import TokenizedRecord
from vramforge_estimator.schemas import Branch, HistogramBin, LengthStats, Objective, RowLength

TOP_ROWS = 10
HISTOGRAM_TARGET_BINS = 32
_QUANTILES = ((50, "p50"), (90, "p90"), (95, "p95"), (99, "p99"))

Getter = Callable[[TokenizedRecord], int | None]


def _pair_max(rec: TokenizedRecord) -> int | None:
    values = [v for v in (rec.chosen_total_tokens, rec.rejected_total_tokens) if v is not None]
    return max(values) if values else None


# Branch series per objective; the first entry is the length fed to the model (primary).
BRANCHES: dict[Objective, tuple[tuple[Branch, Getter], ...]] = {
    Objective.SFT: (
        (Branch.SEQUENCE, lambda r: r.sequence_tokens),
        (Branch.PROMPT, lambda r: r.prompt_tokens),
        (Branch.COMPLETION, lambda r: r.completion_tokens),
        (Branch.LOSS_TOKENS, lambda r: r.loss_token_count),
    ),
    Objective.DPO: (
        (Branch.PAIR_MAX, _pair_max),
        (Branch.PROMPT, lambda r: r.prompt_tokens),
        (Branch.CHOSEN_SEQUENCE, lambda r: r.chosen_total_tokens),
        (Branch.REJECTED_SEQUENCE, lambda r: r.rejected_total_tokens),
        (Branch.CHOSEN, lambda r: r.chosen_completion_tokens),
        (Branch.REJECTED, lambda r: r.rejected_completion_tokens),
    ),
    Objective.GRPO: ((Branch.PROMPT, lambda r: r.prompt_tokens),),
}


def primary_branch(objective: Objective) -> Branch:
    return BRANCHES[objective][0][0]


def branch_values(objective: Objective, rec: TokenizedRecord) -> Iterator[tuple[Branch, int]]:
    for branch, getter in BRANCHES[objective]:
        value = getter(rec)
        if value is not None:
            yield branch, value


def _bin_width(minimum: int) -> int:
    """Smallest power-of-two bin width >= minimum. Power-of-two edges make typical context
    thresholds (limit - budget, multiples of 2^k) land on bin boundaries, so counts above them
    stay exact when read back from the histogram (validation.validate_context)."""
    width = 1
    while width < minimum:
        width *= 2
    return width


class LengthAccumulator:
    """Exact statistics of one branch. Rows must be added in source order."""

    __slots__ = ("_top", "counts", "max_row_id", "max_value", "n", "total")

    def __init__(self) -> None:
        self.counts: dict[int, int] = {}
        self.total = 0
        self.n = 0
        self.max_value: int | None = None
        self.max_row_id: str | None = None
        self._top: list[tuple[int, int, str]] = []  # min-heap of (length, -row_index, row_id)

    def add(self, length: int, row_index: int, row_id: str) -> None:
        self.counts[length] = self.counts.get(length, 0) + 1
        self.total += length
        self.n += 1
        if self.max_value is None or length > self.max_value:
            self.max_value = length
            self.max_row_id = row_id  # first (lowest index) row reaching the maximum
        entry = (length, -row_index, row_id)
        if len(self._top) < TOP_ROWS:
            heapq.heappush(self._top, entry)
        elif entry > self._top[0]:
            heapq.heapreplace(self._top, entry)

    def quantile(self, percent: int) -> int | None:
        """Nearest-rank quantile: the value at rank ceil(percent/100 * n)."""
        if self.n == 0:
            return None
        rank = max(1, (percent * self.n + 99) // 100)
        seen = 0
        for length in sorted(self.counts):
            seen += self.counts[length]
            if seen >= rank:
                return length
        return self.max_value  # pragma: no cover - rank <= n always hits above

    def count_above(self, threshold: int) -> int:
        """Exact number of rows with length > threshold."""
        return sum(c for length, c in self.counts.items() if length > threshold)

    def histogram(self, target_bins: int = HISTOGRAM_TARGET_BINS) -> list[HistogramBin]:
        if self.n == 0:
            return []
        keys = sorted(self.counts)
        lo, hi = keys[0], keys[-1]
        width = _bin_width(-(-(hi - lo + 1) // target_bins))
        start = (lo // width) * width
        bins = [0] * ((hi - start) // width + 1)
        for length in keys:
            bins[(length - start) // width] += self.counts[length]
        return [
            HistogramBin(lo=start + i * width, hi=start + (i + 1) * width, count=count)
            for i, count in enumerate(bins)
        ]

    def stats(self) -> LengthStats:
        if self.n == 0:
            return LengthStats(count=0)
        quantiles = {name: self.quantile(p) for p, name in _QUANTILES}
        return LengthStats(
            count=self.n,
            min=min(self.counts),
            max=self.max_value,
            max_row_id=self.max_row_id,
            mean=self.total / self.n,
            total_tokens=self.total,
            quantiles_exact=True,
            histogram=self.histogram(),
            **quantiles,
        )

    def top_rows(self) -> list[RowLength]:
        return [
            RowLength(row_id=row_id, length=length)
            for length, _neg_index, row_id in sorted(self._top, reverse=True)
        ]

    # ------------------------------------------------------------------ checkpoint state

    def to_state(self) -> dict[str, Any]:
        return {
            "counts": [[length, count] for length, count in sorted(self.counts.items())],
            "total": self.total,
            "n": self.n,
            "max_value": self.max_value,
            "max_row_id": self.max_row_id,
            "top": [[length, -neg, row_id] for length, neg, row_id in self._top],
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> LengthAccumulator:
        acc = cls()
        acc.counts = {int(length): int(count) for length, count in state["counts"]}
        acc.total = int(state["total"])
        acc.n = int(state["n"])
        acc.max_value = state["max_value"]
        acc.max_row_id = state["max_row_id"]
        acc._top = [
            (int(length), -int(index), str(row_id)) for length, index, row_id in state["top"]
        ]
        heapq.heapify(acc._top)
        return acc


__all__ = [
    "BRANCHES",
    "TOP_ROWS",
    "LengthAccumulator",
    "branch_values",
    "primary_branch",
]
