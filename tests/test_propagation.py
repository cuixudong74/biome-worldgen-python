"""Propagation convergence: configurable, bounded, and deadlock-free.

Cross-chunk propagation uses synchronous Jacobi iteration (every round
reads the previous round's snapshot), so there are no read/write cycles
and the iteration can never deadlock even on cyclic wind fields or long
chunk chains.
"""

from __future__ import annotations

from biome_worldgen import World, WorldGenConfig
from biome_worldgen.propagation import propagate
from biome_worldgen.chunk import generate_base_chunk
from biome_worldgen.config import RegionParams


def _cfg(**kw) -> WorldGenConfig:
    base = dict(chunk_size=4, max_propagation_rounds=300, convergence_epsilon=1e-12)
    base.update(kw)
    return WorldGenConfig(**base)


def test_convergence_stops_before_cap():
    world = World(seed=3333, config=_cfg())
    res = world.generate_region(0, 0, 3, 3)
    assert res.converged is True
    assert res.rounds < 300
    assert res.max_delta < 1e-12


def test_cap_is_respected():
    world = World(seed=3434, config=_cfg(max_propagation_rounds=3))
    res = world.generate_region(0, 0, 3, 3)
    assert res.rounds <= 3
    # Still finite and in range despite early stop.
    for key in world.chunk_keys():
        for cell in world.get_chunk(*key).cells:
            assert 0.0 <= cell.moisture <= 1.0
            assert 0.0 <= cell.vegetation <= 1.0


def test_disabled_propagation():
    world = World(seed=3535, config=_cfg(propagation_enabled=False))
    res = world.generate_region(0, 0, 2, 2)
    assert res.converged is True
    assert res.rounds == 0
    # Vegetation stays at its base value (0) when propagation is off.
    for key in world.chunk_keys():
        for cell in world.get_chunk(*key).cells:
            assert cell.vegetation == 0.0


def test_cyclic_wind_no_deadlock():
    """A wind field with a cyclic component (dz != 0) must still
    converge -- the Jacobi iteration has no data dependencies within a
    round, so cycles cannot deadlock."""
    world = World(seed=3636, config=_cfg(wind_dx=1, wind_dz=1))
    res = world.generate_region(0, 0, 4, 4)
    assert res.converged is True
    assert res.rounds < 300


def test_long_chain_no_deadlock():
    """A long chain of chunks must converge without deadlock."""
    world = World(seed=3737, config=_cfg())
    res = world.generate_region(0, 0, 9, 0)  # 10 chunks in a row
    assert res.converged is True
    assert res.rounds < 300


def test_propagate_pure_function():
    """propagate() is a pure function of its inputs: calling it twice on
    the same snapshot gives the same result."""
    cfg = _cfg()
    chunks = {}
    for cz in range(3):
        for cx in range(3):
            chunks[(cx, cz)] = generate_base_chunk(3838, cfg, RegionParams(), cx, cz)
    r1 = propagate(cfg, dict(chunks))
    r2 = propagate(cfg, dict(chunks))
    for key in r1.chunks:
        assert r1.chunks[key].cells == r2.chunks[key].cells


def test_unchanged_chunks_keep_identity():
    """propagate() returns the same object for chunks that did not
    change, so callers can detect rewrites by identity."""
    cfg = _cfg()
    chunks = {}
    for cz in range(3):
        for cx in range(3):
            chunks[(cx, cz)] = generate_base_chunk(3939, cfg, RegionParams(), cx, cz)
    result = propagate(cfg, dict(chunks))
    # After a full propagation, re-running should keep all identities.
    result2 = propagate(cfg, result.chunks)
    for key in result.chunks:
        assert result2.chunks[key] is result.chunks[key]
