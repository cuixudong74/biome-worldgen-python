"""Chunk storage: atomic immutable versioned snapshots of chunks.

The store never mutates a :class:`Chunk` in place. Writers build complete
chunks first and install them in a batch under one lock, so readers always
observe either the previous whole version or the new whole version of each
chunk - never a half-updated cell grid. Each installation also bumps a
monotonic store generation used for optimistic concurrency checks.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable

from worldgen.chunk import Chunk
from worldgen.coords import ChunkCoord


class ChunkStore:
    """In-memory map of chunk coord -> current committed :class:`Chunk`."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._chunks: dict[tuple[int, int], Chunk] = {}
        self._store_version = 0

    def get(self, coord: ChunkCoord) -> Chunk | None:
        with self._lock:
            return self._chunks.get(coord.key)

    def get_many(
        self, coords: Iterable[ChunkCoord]
    ) -> dict[ChunkCoord, Chunk | None]:
        with self._lock:
            return {c: self._chunks.get(c.key) for c in coords}

    def current_versions(
        self, coords: Iterable[ChunkCoord]
    ) -> dict[ChunkCoord, int | None]:
        """Version a writer must still see at commit time (CAS guard)."""
        with self._lock:
            out: dict[ChunkCoord, int | None] = {}
            for c in coords:
                chunk = self._chunks.get(c.key)
                out[c] = None if chunk is None else chunk.version
            return out

    def check_versions(
        self, expected: dict[ChunkCoord, int | None]
    ) -> bool:
        with self._lock:
            for coord, version in expected.items():
                chunk = self._chunks.get(coord.key)
                current = None if chunk is None else chunk.version
                if current != version:
                    return False
            return True

    def put_many(
        self, chunks: list[Chunk], expected: dict[ChunkCoord, int | None] | None = None
    ) -> bool:
        """Atomically install a batch.

        If ``expected`` is given and any chunk version drifted, nothing is
        installed and ``False`` is returned (optimistic concurrency). Chunk
        versions inside the batch are assigned contiguously starting at the
        current store generation.
        """
        with self._lock:
            if expected is not None and not self.check_versions(expected):
                return False
            base = self._store_version
            for i, chunk in enumerate(sorted(chunks, key=lambda c: c.coord)):
                versioned = Chunk(
                    coord=chunk.coord,
                    version=base + 1 + i,
                    params_digest=chunk.params_digest,
                    cells=chunk.cells,
                )
                self._chunks[chunk.coord.key] = versioned
            self._store_version = base + len(chunks)
            return True

    def replace_all(
        self, chunks: dict[ChunkCoord, Chunk | None]
    ) -> None:
        """Restore an arbitrary set (used by snapshot rollback)."""
        with self._lock:
            for coord, chunk in chunks.items():
                if chunk is None:
                    self._chunks.pop(coord.key, None)
                else:
                    self._chunks[coord.key] = chunk
            if self._chunks:
                self._store_version = max(
                    c.version for c in self._chunks.values()
                )
            else:
                self._store_version = 0

    def snapshot(self) -> dict[ChunkCoord, Chunk]:
        with self._lock:
            return {
                ChunkCoord(cx, cz): chunk
                for (cx, cz), chunk in self._chunks.items()
            }

    def existing_coords(self) -> set[ChunkCoord]:
        with self._lock:
            return {ChunkCoord(cx, cz) for cx, cz in self._chunks}

    @property
    def store_version(self) -> int:
        with self._lock:
            return self._store_version

    def __len__(self) -> int:
        with self._lock:
            return len(self._chunks)
