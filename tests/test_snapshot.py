"""Requirement 5: snapshots and rollback; committed vs unfinished states."""

from __future__ import annotations

import pytest

from worldgen import ChunkCoord, World
from worldgen.config import RegionParams
from worldgen.errors import SnapshotError
from worldgen.types import SnapshotState

RECT = (0, 0, 4, 4)
ALL = [ChunkCoord(x, z) for x in range(4) for z in range(4)]
REGION = (1, 1, 3, 3)
PARAMS = RegionParams(temperature_bias=0.3)


def test_snapshot_captures_and_restores(world):
    world.generate(rect=RECT)
    snap = world.snapshot("baseline")
    assert snap.is_committed
    assert snap.state is SnapshotState.COMMITTED
    assert len(snap.chunks) == 16

    world.apply_region_params("hot", REGION, PARAMS)
    assert world._params.digest() != _snapshot_registry_digest(snap)

    world.rollback(snap.snapshot_id)
    assert world._params.digest() == _snapshot_registry_digest(snap)
    for c in ALL:
        assert world.get_chunk(c).cells == _snap_chunk(snap, c).cells


def test_continue_after_rollback_matches_direct(world, config):
    world.generate(rect=RECT)
    snap = world.snapshot("v1")

    # Take a detour that is later discarded.
    world.apply_region_params("detour", (0, 0, 4, 4),
                              RegionParams(temperature_bias=0.5))
    world.rollback(snap.snapshot_id)

    # Continue from the restored version.
    world.apply_region_params("hot", REGION, PARAMS)

    # Direct path: build baseline, then apply the same region.
    direct = World(config)
    direct.generate(rect=RECT)
    direct.apply_region_params("hot", REGION, PARAMS)

    for c in ALL:
        assert world.get_chunk(c).cells == direct.get_chunk(c).cells, c


def test_in_progress_snapshot_is_distinct_and_not_rollbackable(world):
    world.generate(rect=RECT)
    wip = world.begin_snapshot("work-in-progress")
    assert wip.state is SnapshotState.IN_PROGRESS
    assert not wip.is_committed
    with pytest.raises(SnapshotError):
        world.rollback(wip.snapshot_id)

    committed = world.commit_snapshot(wip.snapshot_id)
    assert committed.state is SnapshotState.COMMITTED
    assert world.latest_snapshot().snapshot_id == committed.snapshot_id


def test_aborted_snapshot_cannot_be_rollback_target(world):
    world.generate(rect=RECT)
    wip = world.begin_snapshot()
    aborted = world.abort_snapshot(wip.snapshot_id)
    assert aborted.state is SnapshotState.ABORTED
    with pytest.raises(SnapshotError):
        world.rollback(wip.snapshot_id)


def test_new_chunks_removed_by_rollback(world, config):
    world.generate(rect=(0, 0, 2, 2))
    snap = world.snapshot("small")
    world.generate(rect=(0, 0, 4, 4))
    assert len(world) == 16
    world.rollback(snap.snapshot_id)
    assert len(world) == 4
    # Continuing generation after rollback reproduces the full world.
    world.generate(rect=(0, 0, 4, 4))
    ref = World(config)
    ref.generate(rect=(0, 0, 4, 4))
    for c in ALL:
        assert world.get_chunk(c).cells == ref.get_chunk(c).cells


def test_rollback_restores_registry_and_dirty(world):
    world.generate(rect=RECT)
    snap = world.snapshot("clean")
    world.apply_region_params(
        "hot", REGION, PARAMS, auto_recompute=False
    )
    assert len(world.dirty_chunks) > 0
    world.rollback(snap.snapshot_id)
    assert len(world.dirty_chunks) == 0
    # Registry restored: a recompute is a no-op now.
    assert world.recompute() == []


def test_snapshot_listing_order_and_states(world):
    world.generate(rect=(0, 0, 2, 2))
    s1 = world.snapshot("one")
    wip = world.begin_snapshot("w")
    s2 = world.commit_snapshot(wip.snapshot_id)
    listed = world.list_snapshots()
    ids = [s.snapshot_id for s in listed]
    assert ids == [s1.snapshot_id, s2.snapshot_id]
    assert [s.state for s in listed] == [
        SnapshotState.COMMITTED,
        SnapshotState.COMMITTED,
    ]


def _snap_chunk(snap, coord: ChunkCoord):
    for chunk in snap.chunks:
        if chunk.coord == coord:
            return chunk
    raise KeyError(coord)


def _snapshot_registry_digest(snap):
    from worldgen.params import ParamRegistry

    reg = ParamRegistry(seed=20240917)
    reg.restore(snap.regions)
    return reg.digest()
