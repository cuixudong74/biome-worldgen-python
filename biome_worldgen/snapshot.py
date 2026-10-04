"""World snapshots and versioned rollback."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from .chunk import Chunk


class SnapshotState(str, Enum):
    """Lifecycle state of a snapshot.

    ``PENDING`` snapshots have been opened but not yet committed;
    ``COMMITTED`` snapshots are immutable and may be rolled back to;
    ``ABORTED`` snapshots were cancelled and can never be restored.
    """

    PENDING = "pending"
    COMMITTED = "committed"
    ABORTED = "aborted"


@dataclass(frozen=True)
class Snapshot:
    """An immutable, consistent copy of the world's chunk map at a version."""

    id: str
    label: str
    state: SnapshotState
    version: int
    parent_id: str | None
    created_at: float
    chunks: Mapping[tuple[int, int], Chunk] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def chunk_keys(self) -> tuple[tuple[int, int], ...]:
        return tuple(self.chunks.keys())
