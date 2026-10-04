"""Per-chunk lock table with ordered acquisition to prevent deadlock.

Locks are always acquired in canonical chunk order (cx, then cz), so even
if two requests want each other's chunks there is a single global ordering
and no hold-and-wait cycle can deadlock. A bounded acquisition timeout
guards against pathological scheduling.

The locks are plain :class:`threading.Lock` (non-reentrant); a generation
transaction acquires each chunk lock exactly once for its whole lifetime
and releases in reverse order.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable

from worldgen.coords import ChunkCoord
from worldgen.errors import LockAcquireTimeout


class LockTable:
    """Hands out one lock per chunk coordinate, created on demand."""

    def __init__(self, timeout: float = 30.0) -> None:
        self._guard = threading.Lock()
        self._locks: dict[tuple[int, int], threading.Lock] = {}
        self._timeout = timeout

    def _get(self, coord: ChunkCoord) -> threading.Lock:
        key = coord.key
        with self._guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._locks[key] = lock
            return lock

    def acquire_all(self, coords: Iterable[ChunkCoord]) -> list:
        """Acquire locks in canonical (sorted) order; raise on timeout."""
        ordered = sorted(set(coords))
        acquired: list = []
        for coord in ordered:
            lock = self._get(coord)
            if not lock.acquire(timeout=self._timeout):
                self.release_all(acquired)
                raise LockAcquireTimeout(
                    f"timed out acquiring lock for chunk {coord}"
                )
            acquired.append(lock)
        return acquired

    def release_all(self, locks: Iterable) -> None:
        for lock in reversed(list(locks)):
            lock.release()
