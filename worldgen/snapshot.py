"""World snapshots: committed, in-progress and aborted states.

A snapshot captures an immutable point-in-time copy of every committed
chunk and the whole region registry. Snapshots are append-only and their
state is explicit:

* ``IN_PROGRESS`` - created by :meth:`SnapshotBook.begin`; not yet a
  consistent recovery point (the capture is still being assembled).
* ``COMMITTED`` - a complete, consistent snapshot that can be rolled back
  to and that future generation can continue from.
* ``ABORTED`` - an in-progress capture that was discarded; rollback to it
  is rejected.

Rolling back to a committed snapshot restores both chunks and parameters,
so continuing to generate afterwards yields the same cells as generating
directly on that historical version.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from collections.abc import Iterable

from worldgen.chunk import Chunk
from worldgen.params import Region
from worldgen.rng import mix64
from worldgen.types import SnapshotState


@dataclasses.dataclass(frozen=True, slots=True)
class Snapshot:
    snapshot_id: str
    state: SnapshotState
    created_at: float
    committed_at: float | None
    chunks: tuple[Chunk, ...]
    regions: tuple[Region, ...]
    next_chunk_version: int
    note: str = ""

    @property
    def is_committed(self) -> bool:
        return self.state is SnapshotState.COMMITTED


class SnapshotBook:
    """Append-only, thread-safe snapshot log with rollback support."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._snapshots: dict[str, Snapshot] = {}
        self._order: list[str] = []
        self._counter = 0

    def _new_id(self) -> str:
        self._counter += 1
        raw = mix64(int(time.time_ns() & 0xFFFFFFFFFFFFFFFF), self._counter)
        return f"snap-{self._counter:06d}-{raw:016x}"

    def begin(
        self,
        chunks: Iterable[Chunk] = (),
        regions: Iterable[Region] = (),
        next_chunk_version: int = 0,
        note: str = "",
    ) -> Snapshot:
        """Record an in-progress snapshot (capture may still be filling)."""
        with self._lock:
            snap = Snapshot(
                snapshot_id=self._new_id(),
                state=SnapshotState.IN_PROGRESS,
                created_at=time.time(),
                committed_at=None,
                chunks=tuple(chunks),
                regions=tuple(regions),
                next_chunk_version=next_chunk_version,
                note=note,
            )
            self._snapshots[snap.snapshot_id] = snap
            self._order.append(snap.snapshot_id)
            return snap

    def _replace(self, snap: Snapshot) -> Snapshot:
        with self._lock:
            self._snapshots[snap.snapshot_id] = snap
            return snap

    def commit(
        self,
        snapshot_id: str,
        chunks: Iterable[Chunk] | None = None,
        regions: Iterable[Region] | None = None,
        next_chunk_version: int | None = None,
    ) -> Snapshot:
        """Promote an in-progress snapshot to committed.

        The complete capture is supplied here (or completed at begin time);
        an in-progress entry never serves as a recovery point.
        """
        with self._lock:
            snap = self._snapshots[snapshot_id]
            committed = dataclasses.replace(
                snap,
                state=SnapshotState.COMMITTED,
                committed_at=time.time(),
                chunks=snap.chunks if chunks is None else tuple(chunks),
                regions=snap.regions if regions is None else tuple(regions),
                next_chunk_version=(
                    snap.next_chunk_version
                    if next_chunk_version is None
                    else next_chunk_version
                ),
            )
            self._snapshots[snapshot_id] = committed
            return committed

    def abort(self, snapshot_id: str) -> Snapshot:
        with self._lock:
            snap = self._snapshots[snapshot_id]
            aborted = dataclasses.replace(snap, state=SnapshotState.ABORTED)
            self._snapshots[snapshot_id] = aborted
            return aborted

    def get(self, snapshot_id: str) -> Snapshot:
        with self._lock:
            return self._snapshots[snapshot_id]

    def latest_committed(self) -> Snapshot | None:
        with self._lock:
            for sid in reversed(self._order):
                snap = self._snapshots[sid]
                if snap.state is SnapshotState.COMMITTED:
                    return snap
            return None

    def list_snapshots(self) -> tuple[Snapshot, ...]:
        with self._lock:
            return tuple(self._snapshots[sid] for sid in self._order)
