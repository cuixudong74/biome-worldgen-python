"""World facade: generation, incremental recompute, concurrency, snapshots.

Concurrency model
-----------------
* Every committed chunk is immutable; readers never take locks and always
  see a complete version.
* A write transaction takes the per-chunk locks of its *write set* in
  canonical order (deadlock-free) and performs an optimistic version check
  at commit (CAS). If a raced writer committed first, the deterministic
  computation is simply repeated; results are identical, and the world
  converges without cross-writing chunks.
* Cancellation/timeout is checked at phase boundaries. A pass that aborts
  before ``put_many`` installs nothing, so no partial writes remain.

Incremental recompute
---------------------
Parameter changes mark intersecting chunks; the write set is expanded by
``affected_chunk_radius`` and intersected with existing chunks (unaffected
and not-yet-generated chunks are never rewritten). The pure pipeline then
guarantees the written cells equal a full rebuild.
"""

from __future__ import annotations

import threading

from worldgen.cancellation import CancelToken
from worldgen.chunk import Chunk
from worldgen.config import RegionParams, WorldConfig
from worldgen.coords import ChunkCoord, chunks_in_rect
from worldgen.errors import ConcurrentModificationError
from worldgen.locks import LockTable
from worldgen.params import ParamRegistry, Region
from worldgen.pipeline import affected_chunk_radius, affected_chunks, run_pass
from worldgen.snapshot import Snapshot, SnapshotBook
from worldgen.store import ChunkStore
from worldgen.types import SnapshotState


class World:
    """A versioned, concurrently accessible procedural world."""

    def __init__(self, config: WorldConfig | None = None) -> None:
        self.config = config or WorldConfig()
        self._store = ChunkStore()
        self._params = ParamRegistry(seed=self.config.seed)
        self._locks = LockTable(timeout=self.config.lock_timeout_seconds)
        self._snapshots = SnapshotBook()
        # Chunks whose inputs changed and which still need recomputation.
        self._dirty: set[ChunkCoord] = set()
        # Serialises registry edits + dirty bookkeeping + snapshots.
        self._meta_lock = threading.RLock()

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def _write_transaction(
        self, write_coords: set[ChunkCoord], cancel: CancelToken | None
    ) -> list[Chunk]:
        """Lock -> deterministic compute -> CAS commit, with retries."""
        if not write_coords:
            return []
        ordered = sorted(write_coords)
        locks = self._locks.acquire_all(ordered)
        try:
            for attempt in range(self.config.max_commit_retries):
                if cancel is not None:
                    cancel.check()
                expected = self._store.current_versions(ordered)
                digest = self._params.digest()
                # Pure computation: identical on every retry.
                chunks = run_pass(
                    write_coords=write_coords,
                    config=self.config,
                    resolver=self._params,
                    params_digest=digest,
                    cancel=cancel,
                )
                if cancel is not None:
                    cancel.check()
                # Drop chunks whose content is unchanged, so unaffected
                # chunks keep their old version and are never rewritten.
                changed: list[Chunk] = []
                for chunk in chunks:
                    current = self._store.get(chunk.coord)
                    if current is None or current.cells != chunk.cells:
                        changed.append(chunk)
                changed_coords = {c.coord for c in changed}
                changed_expected = {
                    coord: version
                    for coord, version in expected.items()
                    if coord in changed_coords
                }
                if self._store.put_many(changed, expected=changed_expected):
                    with self._meta_lock:
                        self._dirty -= set(ordered)
                    return [self._store.get(c.coord) for c in changed]
            raise ConcurrentModificationError(
                "exhausted commit retries due to concurrent writers"
            )
        finally:
            self._locks.release_all(locks)

    def generate(
        self,
        coords: set[ChunkCoord] | None = None,
        rect: tuple[int, int, int, int] | None = None,
        *,
        cancel: CancelToken | None = None,
    ) -> list[Chunk]:
        """Generate a set of chunks (or a chunk-grid rectangle)."""
        if coords is None and rect is None:
            raise ValueError("provide coords or rect")
        targets = set(coords) if coords is not None else set(
            chunks_in_rect(*rect)
        )
        if not targets:
            raise ValueError("generate target set is empty")
        token = self._deadline(cancel)
        return self._write_transaction(targets, token)

    def get_chunk(self, coord: ChunkCoord) -> Chunk | None:
        return self._store.get(coord)

    # ------------------------------------------------------------------
    # Parameters / incremental recompute
    # ------------------------------------------------------------------
    def _recompute_set(
        self, hints: set[ChunkCoord] | None, cancel: CancelToken | None
    ) -> list[ChunkCoord]:
        radius = affected_chunk_radius(self.config.max_propagation_rounds)
        with self._meta_lock:
            if hints is not None:
                self._dirty |= set(hints)
            dirty = set(self._dirty)
        affected = affected_chunks(dirty, radius)
        # Only existing chunks can be rewritten; others will be correct if
        # generated later from the same base fields.
        return sorted(affected & self._store.existing_coords())

    def apply_region_params(
        self,
        region_id: str,
        rect: tuple[int, int, int, int],
        params: RegionParams,
        *,
        auto_recompute: bool = True,
        cancel: CancelToken | None = None,
    ) -> list[Chunk]:
        region = Region(region_id, rect[0], rect[1], rect[2], rect[3], params)
        with self._meta_lock:
            touched = self._params.upsert_region(region)
            self._dirty |= touched
        if not auto_recompute:
            return []
        return self.recompute(cancel=cancel)

    def remove_region(
        self,
        region_id: str,
        *,
        auto_recompute: bool = True,
        cancel: CancelToken | None = None,
    ) -> list[Chunk]:
        with self._meta_lock:
            touched = self._params.remove_region(region_id)
            self._dirty |= touched
        if not auto_recompute:
            return []
        return self.recompute(cancel=cancel)

    def recompute(
        self,
        hints: set[ChunkCoord] | None = None,
        *,
        cancel: CancelToken | None = None,
    ) -> list[Chunk]:
        """Incrementally recompute chunks affected by parameter changes."""
        token = self._deadline(cancel)
        targets = self._recompute_set(hints, token)
        if not targets:
            return []
        return self._write_transaction(set(targets), token)

    @property
    def dirty_chunks(self) -> frozenset[ChunkCoord]:
        with self._meta_lock:
            return frozenset(self._dirty)

    # ------------------------------------------------------------------
    # Snapshots / rollback
    # ------------------------------------------------------------------
    def snapshot(self, note: str = "") -> Snapshot:
        """Capture a *committed* snapshot of chunks and parameters."""
        with self._meta_lock:
            snap = self._snapshots.begin(note=note)
            chunks = tuple(self._store.snapshot().values())
            regions = self._params.state_snapshot()
            return self._snapshots.commit(
                snap.snapshot_id,
                chunks=chunks,
                regions=regions,
                next_chunk_version=self._store.store_version,
            )

    def begin_snapshot(self, note: str = "") -> Snapshot:
        """Create an explicitly in-progress (not yet consistent) snapshot."""
        with self._meta_lock:
            return self._snapshots.begin(note=note)

    def commit_snapshot(self, snapshot_id: str) -> Snapshot:
        with self._meta_lock:
            snap = self._snapshots.get(snapshot_id)
            if snap.state is not SnapshotState.IN_PROGRESS:
                from worldgen.errors import SnapshotError

                raise SnapshotError(
                    f"snapshot {snapshot_id} is {snap.state.value}"
                )
            chunks = tuple(self._store.snapshot().values())
            regions = self._params.state_snapshot()
            return self._snapshots.commit(
                snapshot_id,
                chunks=chunks,
                regions=regions,
                next_chunk_version=self._store.store_version,
            )

    def abort_snapshot(self, snapshot_id: str) -> Snapshot:
        with self._meta_lock:
            return self._snapshots.abort(snapshot_id)

    def rollback(self, snapshot_id: str) -> None:
        """Restore chunks + parameters to a committed snapshot.

        All chunk locks of the current world are taken first, so a rollback
        cannot interleave with an in-flight write; aborted/in-progress
        snapshots are rejected.
        """
        with self._meta_lock:
            snap = self._snapshots.get(snapshot_id)
            if snap.state is not SnapshotState.COMMITTED:
                from worldgen.errors import SnapshotError

                raise SnapshotError(
                    f"cannot roll back to {snap.state.value} snapshot"
                )
            coords = set(self._store.existing_coords()) | {
                c.coord for c in snap.chunks
            }
        locks = self._locks.acquire_all(coords)
        try:
            with self._meta_lock:
                target = {c.coord: c for c in snap.chunks}
                self._store.replace_all(
                    {coord: target.get(coord) for coord in coords}
                )
                self._params.restore(snap.regions)
                self._dirty.clear()
        finally:
            self._locks.release_all(locks)

    def list_snapshots(self) -> tuple[Snapshot, ...]:
        return self._snapshots.list_snapshots()

    def latest_snapshot(self) -> Snapshot | None:
        return self._snapshots.latest_committed()

    # ------------------------------------------------------------------
    def _deadline(self, cancel: CancelToken | None) -> CancelToken:
        if cancel is not None:
            return cancel
        return CancelToken.with_timeout(self.config.generation_timeout_seconds)

    def __len__(self) -> int:
        return len(self._store)
