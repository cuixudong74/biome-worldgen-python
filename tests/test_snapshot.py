"""World snapshots, versioned rollback and resumable generation."""

from __future__ import annotations

import pytest

from biome_worldgen import SnapshotState, World, WorldGenConfig


def _cfg(**kw) -> WorldGenConfig:
    base = dict(chunk_size=4, max_propagation_rounds=300, convergence_epsilon=1e-12)
    base.update(kw)
    return WorldGenConfig(**base)


def _fingerprint(world: World) -> dict:
    out = {}
    for key in sorted(world.chunk_keys()):
        chunk = world.get_chunk(*key)
        out[key] = tuple(
            (c.height, c.temperature, c.moisture, c.vegetation) for c in chunk.cells
        )
    return out


def test_snapshot_is_committed():
    world = World(seed=2525, config=_cfg())
    world.generate_region(0, 0, 2, 2)
    snap = world.create_snapshot(label="v1")
    assert snap.state is SnapshotState.COMMITTED
    assert world.current_snapshot_id == snap.id
    assert world.get_snapshot(snap.id).state is SnapshotState.COMMITTED


def test_pending_snapshot_state():
    world = World(seed=2626, config=_cfg())
    world.generate_region(0, 0, 1, 1)
    pending = world.begin_snapshot(label="wip")
    assert pending.state is SnapshotState.PENDING
    # Pending snapshots are visible in the list.
    states = {s.id: s.state for s in world.list_snapshots()}
    assert states[pending.id] is SnapshotState.PENDING
    # Commit transitions to committed.
    committed = world.commit_snapshot(pending.id)
    assert committed.state is SnapshotState.COMMITTED
    assert world.current_snapshot_id == committed.id


def test_abort_pending_snapshot():
    world = World(seed=2727, config=_cfg())
    pending = world.begin_snapshot(label="wip")
    aborted = world.abort_snapshot(pending.id)
    assert aborted.state is SnapshotState.ABORTED
    # Cannot roll back to an aborted snapshot.
    with pytest.raises(ValueError):
        world.rollback(aborted.id)


def test_cannot_rollback_to_pending():
    world = World(seed=2828, config=_cfg())
    pending = world.begin_snapshot()
    with pytest.raises(ValueError):
        world.rollback(pending.id)


def test_rollback_restores_exact_state():
    world = World(seed=2929, config=_cfg())
    world.generate_region(0, 0, 2, 2)
    snap = world.create_snapshot(label="v1")
    before = _fingerprint(world)
    # Continue generating.
    world.generate_region(3, 0, 4, 2)
    assert _fingerprint(world) != before
    # Roll back.
    world.rollback(snap.id)
    assert _fingerprint(world) == before
    assert world.current_snapshot_id == snap.id


def test_continue_after_rollback_matches_direct():
    """After rollback to version V, continuing generation must match
    generating directly on version V."""
    # Path A: generate A, snapshot, generate B, rollback, generate B again.
    wA = World(seed=3030, config=_cfg())
    wA.generate_region(0, 0, 2, 2)
    snap = wA.create_snapshot(label="after-A")
    wA.generate_region(3, 0, 4, 2)
    wA.rollback(snap.id)
    wA.generate_region(3, 0, 4, 2)

    # Path B: generate A then B in one go (direct).
    wB = World(seed=3030, config=_cfg())
    wB.generate_region(0, 0, 4, 2)

    assert _fingerprint(wA) == _fingerprint(wB)


def test_rollback_then_regenerate_is_stable():
    world = World(seed=3131, config=_cfg())
    world.generate_region(0, 0, 2, 2)
    snap = world.create_snapshot()
    world.generate_region(0, 0, 3, 3)  # expand to 4x4
    world.rollback(snap.id)
    # Re-generate the same region that was rolled back.
    world.generate_region(0, 0, 3, 3)
    ref = World(seed=3131, config=_cfg())
    ref.generate_region(0, 0, 3, 3)
    assert _fingerprint(world) == _fingerprint(ref)


def test_snapshot_preserves_immutable_chunks():
    world = World(seed=3232, config=_cfg())
    world.generate_region(0, 0, 1, 1)
    snap = world.create_snapshot()
    original = world.get_chunk(0, 0)
    # Generate more; the snapshot's chunk must be the same object.
    world.generate_region(2, 2, 3, 3)
    assert snap.chunks[(0, 0)] is original
