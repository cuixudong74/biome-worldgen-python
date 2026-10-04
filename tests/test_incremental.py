"""Requirement 3: incremental recompute == full rebuild; scoped writes;
bounded/loop-free propagation with configurable stopping."""

from __future__ import annotations

import pytest

from worldgen import ChunkCoord, WorldConfig
from worldgen.config import RegionParams

RECT = (0, 0, 4, 4)
ALL = [ChunkCoord(x, z) for x in range(4) for z in range(4)]
REGION_RECT = (1, 1, 3, 3)  # 2x2 central chunk region


def _full_rebuild_reference(config, region_rect, region_params, region_id="hot"):
    """Fresh world where the region exists *before* any chunk is made."""
    from worldgen import World

    ref = World(config)
    ref.apply_region_params(
        region_id, region_rect, region_params, auto_recompute=False
    )
    ref.generate(rect=RECT)
    return ref


def test_incremental_equals_full_rebuild(world, config):
    world.generate(rect=RECT)
    world.apply_region_params("hot", REGION_RECT, RegionParams(temperature_bias=0.3))
    ref = _full_rebuild_reference(config, REGION_RECT, RegionParams(temperature_bias=0.3))
    for coord in ALL:
        assert world.get_chunk(coord).cells == ref.get_chunk(coord).cells, coord


def test_unaffected_chunks_not_rewritten(config):
    # 6x6 world with a central 2x2 region: outer corner chunks are two
    # chunks away from the region (Chebyshev distance 2 > influence radius 1)
    # and must keep both their version and their content.
    from worldgen import World

    world = World(config)
    world.generate(rect=(0, 0, 6, 6))
    all6 = [ChunkCoord(x, z) for x in range(6) for z in range(6)]
    versions = {c: world.get_chunk(c).version for c in all6}
    snapshot_cells = {c: world.get_chunk(c).cells for c in all6}
    world.apply_region_params(
        "hot", (2, 2, 4, 4), RegionParams(moisture_bias=0.25)
    )
    far_corners = [
        ChunkCoord(0, 0), ChunkCoord(5, 0),
        ChunkCoord(0, 5), ChunkCoord(5, 5),
    ]
    for coord in far_corners:
        assert world.get_chunk(coord).version == versions[coord], coord
        assert world.get_chunk(coord).cells == snapshot_cells[coord], coord
    # At least one near-region chunk was actually rewritten.
    near = [c for c in all6 if c not in far_corners]
    assert any(world.get_chunk(c).version != versions[c] for c in near)


def test_chained_changes_still_match_full_rebuild(world, config):
    world.generate(rect=RECT)
    world.apply_region_params("hot", REGION_RECT, RegionParams(temperature_bias=0.3))
    world.apply_region_params("wet", (0, 0, 2, 2), RegionParams(moisture_bias=0.2))
    world.apply_region_params(
        "hot", REGION_RECT, RegionParams(temperature_bias=-0.1)
    )  # edit existing region
    # Reference applies the same final registry, then builds everything.
    from worldgen import World

    ref = World(config)
    ref.apply_region_params("wet", (0, 0, 2, 2), RegionParams(moisture_bias=0.2),
                            auto_recompute=False)
    ref.apply_region_params("hot", REGION_RECT, RegionParams(temperature_bias=-0.1),
                            auto_recompute=False)
    ref.generate(rect=RECT)
    for coord in ALL:
        assert world.get_chunk(coord).cells == ref.get_chunk(coord).cells, coord


def test_remove_region_recomputes_and_matches(world, config):
    world.generate(rect=RECT)
    world.apply_region_params("hot", REGION_RECT, RegionParams(temperature_bias=0.3))
    world.remove_region("hot")
    from worldgen import World

    ref = World(config)
    ref.generate(rect=RECT)
    for coord in ALL:
        assert world.get_chunk(coord).cells == ref.get_chunk(coord).cells, coord


def test_deferred_recompute_does_not_write_on_param_change(world):
    world.generate(rect=RECT)
    versions = {c: world.get_chunk(c).version for c in ALL}
    world.apply_region_params(
        "hot", REGION_RECT, RegionParams(temperature_bias=0.3),
        auto_recompute=False,
    )
    # Nothing is rewritten yet, but chunks are marked dirty.
    for coord in ALL:
        assert world.get_chunk(coord).version == versions[coord]
    assert ChunkCoord(1, 1) in world.dirty_chunks
    changed = world.recompute()
    assert len(changed) > 0
    assert len(world.dirty_chunks) == 0


def test_noop_recompute_writes_nothing(world):
    world.generate(rect=RECT)
    assert world.recompute() == []
    versions = {c: world.get_chunk(c).version for c in ALL}
    # Recompute after an identical re-application changes nothing.
    world.apply_region_params(
        "hot", REGION_RECT, RegionParams(temperature_bias=0.0),
        auto_recompute=False,
    )
    assert world.recompute() == []
    for coord in ALL:
        assert world.get_chunk(coord).version == versions[coord]


def test_propagation_always_terminates_on_cycle(world):
    # A dense world is a large cyclic dependency graph; a bounded run must
    # still return and converge (fixed point) without hanging.
    world.apply_region_params("hot", (0, 0, 3, 3), RegionParams(temperature_bias=0.5),
                              auto_recompute=False)
    world.generate(rect=RECT)
    # Repeated recomputes must be idempotent (fixed point reached).
    world.apply_region_params("again", (1, 1, 2, 2), RegionParams(temperature_bias=0.1))
    v1 = {c: world.get_chunk(c).cells for c in ALL}
    world._dirty.update(ALL)
    world.recompute()
    v2 = {c: world.get_chunk(c).cells for c in ALL}
    assert v1 == v2


def test_max_rounds_is_configurable_and_limits_influence(config):
    cfg_small = WorldConfig(seed=config.seed, max_propagation_rounds=1)
    from worldgen import World

    a = World(cfg_small)
    a.generate(rect=RECT)
    a.apply_region_params("hot", REGION_RECT, RegionParams(temperature_bias=0.4))
    ref = World(cfg_small)
    ref.apply_region_params("hot", REGION_RECT, RegionParams(temperature_bias=0.4),
                            auto_recompute=False)
    ref.generate(rect=RECT)
    for coord in ALL:
        assert a.get_chunk(coord).cells == ref.get_chunk(coord).cells


def test_non_converged_policy_raise(config):
    cfg = WorldConfig(
        seed=config.seed,
        max_propagation_rounds=1,
        convergence_tolerance=1e-12,
        non_converged_policy="raise",
    )
    from worldgen import World
    from worldgen.errors import NonConvergedError

    w = World(cfg)
    w.apply_region_params("hot", REGION_RECT, RegionParams(temperature_bias=0.5),
                          auto_recompute=False)
    with pytest.raises(NonConvergedError):
        w.generate(rect=RECT)
    # Nothing was installed because the pass aborted before materialising.
    assert len(w) == 0
