"""The world: a chunk grid with generation, recompute, snapshots and locks."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable

from .chunk import Chunk, generate_base_chunk
from .concurrency import (
    CancelToken,
    OperationHandle,
    check_deadline,
    deadline_from_timeout,
)
from .config import RegionParams, WorldGenConfig
from .propagation import PropagationResult, propagate
from .snapshot import Snapshot, SnapshotState


@dataclass
class RegionResult:
    added_keys: tuple[tuple[int, int], ...]
    updated_keys: tuple[tuple[int, int], ...]
    rounds: int
    converged: bool
    max_delta: float


@dataclass
class RecomputeResult:
    updated_keys: tuple[tuple[int, int], ...]
    rounds: int
    converged: bool
    max_delta: float


class World:
    """An in-memory, concurrent, versioned world.

    Concurrency model:

    * chunks are immutable value objects; every update swaps a reference,
      so a reader always observes a complete version;
    * :data:`_commit_lock` serialises commits and snapshot captures;
    * per-chunk locks guard reads of individual chunks;
    * heavy computation (base generation, propagation) happens on private
      working copies outside the commit lock.
    """

    def __init__(self, seed: int, config: WorldGenConfig | None = None) -> None:
        self.seed = int(seed)
        self.config = config or WorldGenConfig()
        self._chunks: dict[tuple[int, int], Chunk] = {}
        self._chunk_locks: dict[tuple[int, int], threading.RLock] = {}
        self._commit_lock = threading.RLock()
        # Serialises whole generation / recompute operations so that
        # concurrent requests can never interleave their propagation and
        # commit phases (which would otherwise miss cross-region seams).
        self._generation_lock = threading.RLock()
        self._snapshots: dict[str, Snapshot] = {}
        self._current_snapshot_id: str | None = None
        self._version = 0
        self._ops: dict[str, OperationHandle] = {}
        self._op_lock = threading.RLock()

    # ------------------------------------------------------------------
    # basic reads
    # ------------------------------------------------------------------
    @property
    def version(self) -> int:
        return self._version

    @property
    def current_snapshot_id(self) -> str | None:
        return self._current_snapshot_id

    def chunk_keys(self) -> list[tuple[int, int]]:
        with self._commit_lock:
            return list(self._chunks.keys())

    def has_chunk(self, cx: int, cz: int) -> bool:
        with self._commit_lock:
            return (cx, cz) in self._chunks

    def get_chunk(self, cx: int, cz: int) -> Chunk | None:
        key = (cx, cz)
        with self._commit_lock:
            chunk = self._chunks.get(key)
            lock = self._chunk_locks.setdefault(key, threading.RLock())
        with lock:
            return chunk

    def snapshot_refs(self) -> dict[tuple[int, int], Chunk]:
        """Copy chunk references under the commit lock (consistent view)."""
        with self._commit_lock:
            return dict(self._chunks)

    # ------------------------------------------------------------------
    # generation
    # ------------------------------------------------------------------
    def _generate_base_worker(
        self,
        keys: list[tuple[int, int]],
        params: RegionParams,
        token: CancelToken | None,
        deadline: float | None,
    ) -> dict[tuple[int, int], Chunk]:
        produced: dict[tuple[int, int], Chunk] = {}
        for key in keys:
            if token is not None:
                token.raise_if_cancelled()
            check_deadline(deadline)
            cx, cz = key
            produced[key] = generate_base_chunk(self.seed, self.config, params, cx, cz)
        return produced

    def generate_region(
        self,
        x0: int,
        z0: int,
        x1: int,
        z1: int,
        params: RegionParams | None = None,
        cancel_token: CancelToken | None = None,
        timeout: float | None = None,
    ) -> RegionResult:
        """Generate (or extend) the inclusive chunk rectangle
        ``[x0, x1] * [z0, z1]``.

        Missing chunks are created from the seed; afterwards the whole
        loaded world is relaxed to its cross-chunk fixed point and the
        result is committed atomically.  Cancelling or timing out before
        the commit leaves the world untouched.
        """
        params = params or RegionParams()
        deadline = deadline_from_timeout(timeout)

        with self._generation_lock:
            with self._commit_lock:
                working = dict(self._chunks)

            wanted = [
                (cx, cz)
                for cz in range(min(z0, z1), max(z0, z1) + 1)
                for cx in range(min(x0, x1), max(x0, x1) + 1)
                if (cx, cz) not in working
            ]

            if wanted:
                workers = min(self.config.worker_threads, len(wanted))
                if workers <= 1:
                    produced = self._generate_base_worker(wanted, params, cancel_token, deadline)
                else:
                    produced = self._generate_base_parallel(wanted, params, cancel_token, deadline, workers)
                working.update(produced)

            result = propagate(self.config, working, cancel_token, deadline)
            return self._commit(result, wanted)

    def _generate_base_parallel(
        self,
        keys: list[tuple[int, int]],
        params: RegionParams,
        token: CancelToken | None,
        deadline: float | None,
        workers: int,
    ) -> dict[tuple[int, int], Chunk]:
        from concurrent.futures import ThreadPoolExecutor

        produced: dict[tuple[int, int], Chunk] = {}
        batches: list[list[tuple[int, int]]] = [[] for _ in range(workers)]
        for i, key in enumerate(keys):
            batches[i % workers].append(key)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(self._generate_base_worker, batch, params, token, deadline)
                for batch in batches if batch
            ]
            for fut in futures:
                produced.update(fut.result())
        return produced

    # ------------------------------------------------------------------
    # incremental recompute
    # ------------------------------------------------------------------
    def recompute(
        self,
        changes: dict[tuple[int, int], RegionParams],
        cancel_token: CancelToken | None = None,
        timeout: float | None = None,
    ) -> RecomputeResult:
        """Recompute existing chunks with locally changed parameters.

        Only the named chunks are regenerated from the seed; the whole
        loaded world is then relaxed to its fixed point.  Chunks whose
        final content is unchanged keep their identity (they are never
        rewritten).  The result matches a full rebuild with the changed
        parameters exactly.
        """
        if not changes:
            raise ValueError("recompute requires at least one changed chunk")
        deadline = deadline_from_timeout(timeout)

        with self._generation_lock:
            with self._commit_lock:
                working = dict(self._chunks)
                for key in changes:
                    if key not in working:
                        raise KeyError(f"cannot recompute missing chunk {key}")

            for key, params in changes.items():
                if cancel_token is not None:
                    cancel_token.raise_if_cancelled()
                check_deadline(deadline)
                cx, cz = key
                # Terrain height is seed-derived and therefore unchanged; the
                # surface fields are re-based and re-grown by propagation.
                working[key] = generate_base_chunk(self.seed, self.config, params, cx, cz)

            result = propagate(self.config, working, cancel_token, deadline)
            return self._commit(result, list(changes.keys()))

    # ------------------------------------------------------------------
    # commit
    # ------------------------------------------------------------------
    def _commit(
        self,
        result: PropagationResult,
        added_hint: Iterable[tuple[int, int]],
    ) -> RegionResult:
        added: list[tuple[int, int]] = []
        updated: list[tuple[int, int]] = []
        with self._commit_lock:
            for key, new_chunk in result.chunks.items():
                old = self._chunks.get(key)
                if old is None:
                    self._chunks[key] = new_chunk
                    added.append(key)
                elif old.cells != new_chunk.cells:
                    self._chunks[key] = new_chunk
                    updated.append(key)
                # else: identical content -> keep the existing object
            self._version += 1
        return RegionResult(
            added_keys=tuple(added),
            updated_keys=tuple(updated),
            rounds=result.rounds,
            converged=result.converged,
            max_delta=result.max_delta,
        )

    # ------------------------------------------------------------------
    # snapshots / versions
    # ------------------------------------------------------------------
    def create_snapshot(self, label: str | None = None) -> Snapshot:
        """Capture the current world as a new committed snapshot."""
        with self._commit_lock:
            snap = Snapshot(
                id=uuid.uuid4().hex,
                label=label or f"snapshot-{len(self._snapshots) + 1}",
                state=SnapshotState.COMMITTED,
                version=self._version,
                parent_id=self._current_snapshot_id,
                created_at=time.time(),
                chunks=MappingProxyType(dict(self._chunks)),
            )
            self._snapshots[snap.id] = snap
            self._current_snapshot_id = snap.id
            return snap

    def begin_snapshot(self, label: str | None = None) -> Snapshot:
        """Open a pending snapshot (not yet committed, not rollback-able)."""
        snap = Snapshot(
            id=uuid.uuid4().hex,
            label=label or f"pending-{len(self._snapshots) + 1}",
            state=SnapshotState.PENDING,
            version=self._version,
            parent_id=self._current_snapshot_id,
            created_at=time.time(),
            chunks=MappingProxyType({}),
        )
        with self._op_lock:
            self._snapshots[snap.id] = snap
        return snap

    def commit_snapshot(self, snapshot_id: str) -> Snapshot:
        """Commit a pending snapshot, capturing the current chunk map."""
        with self._commit_lock:
            pending = self._snapshots.get(snapshot_id)
            if pending is None:
                raise KeyError(f"unknown snapshot {snapshot_id}")
            if pending.state is not SnapshotState.PENDING:
                raise ValueError(f"snapshot {snapshot_id} is not pending")
            committed = Snapshot(
                id=pending.id,
                label=pending.label,
                state=SnapshotState.COMMITTED,
                version=self._version,
                parent_id=self._current_snapshot_id,
                created_at=time.time(),
                chunks=MappingProxyType(dict(self._chunks)),
            )
            self._snapshots[snapshot_id] = committed
            self._current_snapshot_id = snapshot_id
            return committed

    def abort_snapshot(self, snapshot_id: str) -> Snapshot:
        """Mark a pending snapshot as aborted (never rollback-able)."""
        with self._op_lock:
            snap = self._snapshots.get(snapshot_id)
            if snap is None:
                raise KeyError(f"unknown snapshot {snapshot_id}")
            if snap.state is not SnapshotState.PENDING:
                raise ValueError(f"snapshot {snapshot_id} is not pending")
            aborted = Snapshot(
                id=snap.id,
                label=snap.label,
                state=SnapshotState.ABORTED,
                version=snap.version,
                parent_id=snap.parent_id,
                created_at=snap.created_at,
                chunks=MappingProxyType({}),
            )
            self._snapshots[snapshot_id] = aborted
            return aborted

    def rollback(self, snapshot_id: str) -> Snapshot:
        """Restore the chunk map of a committed snapshot.

        Continuing generation after a rollback is equivalent to having
        generated directly from that version.
        """
        with self._commit_lock:
            snap = self._snapshots.get(snapshot_id)
            if snap is None:
                raise KeyError(f"unknown snapshot {snapshot_id}")
            if snap.state is not SnapshotState.COMMITTED:
                raise ValueError(
                    f"cannot roll back to snapshot {snapshot_id} "
                    f"in state {snap.state.value}; only committed snapshots "
                    "can be restored"
                )
            self._chunks = dict(snap.chunks)
            self._version += 1
            self._current_snapshot_id = snap.id
            return snap

    def list_snapshots(self) -> list[Snapshot]:
        with self._op_lock:
            return list(self._snapshots.values())

    def get_snapshot(self, snapshot_id: str) -> Snapshot:
        with self._op_lock:
            snap = self._snapshots.get(snapshot_id)
            if snap is None:
                raise KeyError(f"unknown snapshot {snapshot_id}")
            return snap

    # ------------------------------------------------------------------
    # operation registry (used by the API layer)
    # ------------------------------------------------------------------
    def register_operation(self, kind: str) -> OperationHandle:
        handle = OperationHandle(id=uuid.uuid4().hex, kind=kind, token=CancelToken())
        with self._op_lock:
            self._ops[handle.id] = handle
        return handle

    def get_operation(self, operation_id: str) -> OperationHandle:
        with self._op_lock:
            handle = self._ops.get(operation_id)
        if handle is None:
            raise KeyError(f"unknown operation {operation_id}")
        return handle

    def list_operations(self) -> list[OperationHandle]:
        with self._op_lock:
            return list(self._ops.values())
