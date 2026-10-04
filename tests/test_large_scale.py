"""Large-scale generation: many chunks, long propagation chains."""

from __future__ import annotations

from biome_worldgen import World, WorldGenConfig


def test_large_world_generation():
    """A 16x16 chunk world must generate and converge."""
    world = World(seed=4242, config=WorldGenConfig(
        chunk_size=4, max_propagation_rounds=400, convergence_epsilon=1e-12,
    ))
    res = world.generate_region(0, 0, 15, 15)
    assert res.converged is True
    assert len(res.added_keys) == 256
    assert len(world.chunk_keys()) == 256
    # Every cell is finite and in range.
    for key in world.chunk_keys():
        for cell in world.get_chunk(*key).cells:
            assert 0.0 <= cell.moisture <= 1.0
            assert 0.0 <= cell.vegetation <= 1.0


def test_large_world_deterministic():
    """Two large worlds with the same seed must match exactly."""
    def build():
        w = World(seed=4343, config=WorldGenConfig(
            chunk_size=4, max_propagation_rounds=400, convergence_epsilon=1e-12,
        ))
        w.generate_region(0, 0, 11, 11)
        return w

    w1 = build()
    w2 = build()
    for key in w1.chunk_keys():
        assert w1.get_chunk(*key).cells == w2.get_chunk(*key).cells


def test_large_world_incremental_matches_full():
    """On a large world, a local recompute must still match a full rebuild."""
    cfg = WorldGenConfig(chunk_size=4, max_propagation_rounds=400, convergence_epsilon=1e-12)
    full = World(seed=4444, config=cfg)
    full.generate_region(0, 0, 11, 11)
    changed_params = __import__("biome_worldgen").RegionParams(temperature_offset=3.0)
    changed = {(6, 6): changed_params}
    full.recompute({k: (changed_params if k == (6, 6) else __import__("biome_worldgen").RegionParams())
                    for k in full.chunk_keys()})

    inc = World(seed=4444, config=cfg)
    inc.generate_region(0, 0, 11, 11)
    inc.recompute(changed)

    for key in inc.chunk_keys():
        assert inc.get_chunk(*key).cells == full.get_chunk(*key).cells, key
