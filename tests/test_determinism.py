"""Reproducibility: same seed + params -> identical cells, every time."""

from __future__ import annotations

from biome_worldgen import RegionParams, World, WorldGenConfig


def _cfg(**kw) -> WorldGenConfig:
    base = dict(chunk_size=4, max_propagation_rounds=200)
    base.update(kw)
    return WorldGenConfig(**base)


def _fingerprint(world: World) -> list[tuple]:
    """Flat list of every cell field, in deterministic chunk/cell order."""
    out = []
    for key in sorted(world.chunk_keys()):
        chunk = world.get_chunk(*key)
        for cell in chunk.cells:
            out.append((
                cell.height,
                cell.temperature,
                cell.moisture,
                cell.vegetation,
            ))
    return out


def test_same_seed_same_result():
    w1 = World(seed=1234, config=_cfg())
    w2 = World(seed=1234, config=_cfg())
    w1.generate_region(0, 0, 3, 3)
    w2.generate_region(0, 0, 3, 3)
    assert _fingerprint(w1) == _fingerprint(w2)


def test_different_seed_different_result():
    w1 = World(seed=1, config=_cfg())
    w2 = World(seed=2, config=_cfg())
    w1.generate_region(0, 0, 2, 2)
    w2.generate_region(0, 0, 2, 2)
    assert _fingerprint(w1) != _fingerprint(w2)


def test_regeneration_is_stable():
    w = World(seed=7, config=_cfg())
    w.generate_region(0, 0, 2, 2)
    first = _fingerprint(w)
    # Re-running the same generation must not change anything.
    w.generate_region(0, 0, 2, 2)
    assert _fingerprint(w) == first


def test_params_change_result():
    w1 = World(seed=7, config=_cfg())
    w2 = World(seed=7, config=_cfg())
    w1.generate_region(0, 0, 2, 2)
    w2.generate_region(0, 0, 2, 2, params=RegionParams(temperature_offset=5.0))
    assert _fingerprint(w1) != _fingerprint(w2)


def test_history_reproducible():
    """Generate A, snapshot, generate B; a fresh world doing the same
    sequence must produce the same final state."""
    def build(seed: int) -> World:
        w = World(seed=seed, config=_cfg())
        w.generate_region(0, 0, 2, 2)
        w.create_snapshot(label="after-A")
        w.generate_region(3, 0, 4, 2)
        return w

    w1 = build(99)
    w2 = build(99)
    assert _fingerprint(w1) == _fingerprint(w2)


def test_independent_of_generation_timing():
    """Two worlds generated with the same seed but different chunk
    ordering must agree on every cell (propagation is order independent)."""
    w1 = World(seed=55, config=_cfg())
    w2 = World(seed=55, config=_cfg())
    # w1: one batch
    w1.generate_region(0, 0, 3, 3)
    # w2: column by column
    for cx in range(4):
        w2.generate_region(cx, 0, cx, 3)
    assert _fingerprint(w1) == _fingerprint(w2)
