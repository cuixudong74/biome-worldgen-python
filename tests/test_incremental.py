"""Incremental recomputation: local changes converge to the full-rebuild
fixed point, and untouched chunks are never rewritten."""

from __future__ import annotations

from biome_worldgen import RegionParams, World, WorldGenConfig


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


def test_incremental_matches_full_rebuild():
    changed_params = RegionParams(temperature_offset=4.0, moisture_offset=0.2)

    # Full rebuild: regenerate EVERY chunk's base (the changed chunk gets
    # new params, the rest keep defaults), then propagate over the world.
    full = World(seed=707, config=_cfg())
    full.generate_region(0, 0, 5, 5)
    all_keys = full.chunk_keys()
    full.recompute({
        k: (changed_params if k == (2, 2) else RegionParams())
        for k in all_keys
    })

    # Incremental: regenerate only the changed chunk's base, then propagate.
    inc = World(seed=707, config=_cfg())
    inc.generate_region(0, 0, 5, 5)
    inc.recompute({(2, 2): changed_params})

    assert _fingerprint(full) == _fingerprint(inc)


def test_unaffected_chunks_not_rewritten():
    inc = World(seed=808, config=_cfg())
    inc.generate_region(0, 0, 6, 6)
    before = {k: inc.get_chunk(*k) for k in inc.chunk_keys()}

    changed = {(3, 3): RegionParams(temperature_offset=6.0)}
    inc.recompute(changed)

    after = {k: inc.get_chunk(*k) for k in inc.chunk_keys()}
    # Chunks far from the change must keep the exact same object.
    far = [(0, 0), (0, 6), (6, 0), (6, 6), (0, 3), (6, 3)]
    for key in far:
        assert key in after
        assert after[key] is before[key], f"chunk {key} was rewritten"
    # The changed chunk must be a new object.
    assert after[(3, 3)] is not before[(3, 3)]


def test_affected_region_updated():
    inc = World(seed=909, config=_cfg())
    inc.generate_region(0, 0, 4, 4)
    before = {k: inc.get_chunk(*k) for k in inc.chunk_keys()}
    changed = {(2, 2): RegionParams(temperature_offset=8.0)}
    res = inc.recompute(changed)
    after = {k: inc.get_chunk(*k) for k in inc.chunk_keys()}
    assert (2, 2) in res.updated_keys
    assert after[(2, 2)] is not before[(2, 2)]
    # The centre chunk actually warmed up.
    assert after[(2, 2)].cell(3, 3).temperature > before[(2, 2)].cell(3, 3).temperature


def test_recompute_missing_chunk_raises():
    w = World(seed=1010, config=_cfg())
    w.generate_region(0, 0, 2, 2)
    try:
        w.recompute({(5, 5): RegionParams()})
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError for missing chunk")


def test_recompute_empty_raises():
    w = World(seed=1111, config=_cfg())
    w.generate_region(0, 0, 1, 1)
    try:
        w.recompute({})
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for empty changes")


def test_propagation_converges_within_cap():
    """Even with a tight round cap the iteration must terminate (no
    deadlock) and report its convergence status."""
    w = World(seed=1212, config=_cfg(max_propagation_rounds=5))
    w.generate_region(0, 0, 3, 3)
    res = w.generate_region(0, 0, 3, 3)
    assert res.rounds <= 5
    # Values are still finite and in range.
    for key in w.chunk_keys():
        for cell in w.get_chunk(*key).cells:
            assert 0.0 <= cell.moisture <= 1.0
            assert 0.0 <= cell.vegetation <= 1.0


def test_repeated_recompute_idempotent():
    inc = World(seed=1313, config=_cfg())
    inc.generate_region(0, 0, 4, 4)
    changed = {(2, 2): RegionParams(temperature_offset=3.0)}
    inc.recompute(changed)
    snap1 = _fingerprint(inc)
    inc.recompute(changed)
    snap2 = _fingerprint(inc)
    assert snap1 == snap2
