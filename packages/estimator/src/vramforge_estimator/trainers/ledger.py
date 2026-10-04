"""Pure helpers over a shape ledger: timepoint expansion and alias-group contributions.

Shared by the trainer adapters (to size per-timepoint allocator slack) and by the memory engine
(to evaluate timepoints), so both use the same liveness semantics
(docs/methodology.md#peak-evaluation). Lives in `trainers` because `memory.engine` already imports
this package; the reverse import would create a cycle.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from vramforge_estimator.schemas import AllocationSpec, Timepoint

WILDCARD = ":*"


def is_unknown(spec: AllocationSpec) -> bool:
    return spec.bytes_low is None or spec.bytes_high is None


def expand_live_at(entries: Iterable[str], timepoints: Sequence[Timepoint]) -> list[str]:
    """Resolve `live_at` entries (ids or "<PHASE>:*") to timepoint ids in schedule order.

    Unknown ids raise ValueError: a dangling reference is a trainer-adapter bug, not a size.
    """
    ids = {tp.id for tp in timepoints}
    wanted: set[str] = set()
    for entry in entries:
        if entry.endswith(WILDCARD):
            phase = entry[: -len(WILDCARD)]
            wanted.update(tp.id for tp in timepoints if tp.phase.value == phase)
        elif entry in ids:
            wanted.add(entry)
        else:
            raise ValueError(f"live_at references unknown timepoint {entry!r}")
    return [tp.id for tp in sorted(timepoints, key=lambda t: t.order) if tp.id in wanted]


def alive_by_timepoint(
    timepoints: Sequence[Timepoint], allocations: Iterable[AllocationSpec]
) -> dict[str, list[AllocationSpec]]:
    alive: dict[str, list[AllocationSpec]] = {tp.id: [] for tp in timepoints}
    for spec in allocations:
        for tp_id in expand_live_at(spec.live_at, timepoints):
            alive[tp_id].append(spec)
    return alive


@dataclass(frozen=True)
class Contribution:
    """What one storage contributes at a timepoint (an alias group counts once)."""

    spec: AllocationSpec  # representative: the largest member (or the unknown one)
    bytes_low: int | None
    bytes_high: int | None
    aliases: tuple[str, ...] = ()  # other alive members sharing the storage

    @property
    def unknown(self) -> bool:
        return self.bytes_low is None or self.bytes_high is None


def contributions(alive: Iterable[AllocationSpec]) -> list[Contribution]:
    """Collapse alias groups: a group contributes max(low) and max(high) of its alive members;
    any unknown member makes the whole group unknown."""
    out: list[Contribution] = []
    groups: dict[str, list[AllocationSpec]] = {}
    order: list[str | AllocationSpec] = []
    for spec in alive:
        if spec.storage_alias_group is None:
            order.append(spec)
            continue
        if spec.storage_alias_group not in groups:
            groups[spec.storage_alias_group] = []
            order.append(spec.storage_alias_group)
        groups[spec.storage_alias_group].append(spec)
    for item in order:
        if isinstance(item, AllocationSpec):
            out.append(Contribution(item, item.bytes_low, item.bytes_high))
            continue
        members = groups[item]
        unknown = [m for m in members if is_unknown(m)]
        if unknown:
            rep = unknown[0]
            low = high = None
        else:
            rep = max(members, key=lambda m: (m.bytes_high or 0, m.bytes_low or 0))
            low = max(m.bytes_low or 0 for m in members)
            high = max(m.bytes_high or 0 for m in members)
        aliases = tuple(m.name for m in members if m is not rep)
        out.append(Contribution(rep, low, high, aliases))
    return out


def sum_known(values: Iterable[int | None]) -> int | None:
    total = 0
    for value in values:
        if value is None:
            return None
        total += value
    return total
