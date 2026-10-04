"""Requirement 1: seed+params => cell-for-cell identical, repeatable."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from worldgen import ChunkCoord
from worldgen.rng import RNG
from worldgen.terrain import fbm2, value_noise2

RECT = (0, 0, 4, 4)
COORDS = [ChunkCoord(x, z) for x in range(4) for z in range(4)]


def _digest_world(world):
    import hashlib

    h = hashlib.sha256()
    for coord in COORDS:
        for cell in world.get_chunk(coord).cells:
            h.update(repr(cell).encode())
    return h.hexdigest()


def test_same_seed_same_world(world, make_world):
    world.generate(rect=RECT)
    other = make_world()
    other.generate(rect=RECT)
    assert _digest_world(world) == _digest_world(other)


def test_repeated_generation_is_stable(world):
    world.generate(rect=RECT)
    first = _digest_world(world)
    # Regenerating the same region with same inputs yields identical cells.
    world.generate(rect=RECT)
    assert _digest_world(world) == first


def test_different_seed_differs(make_world):
    from worldgen import World, WorldConfig

    a = World(WorldConfig(seed=1))
    b = World(WorldConfig(seed=2))
    a.generate(rect=RECT)
    b.generate(rect=RECT)
    assert _digest_world(a) != _digest_world(b)


def test_generation_order_independence(make_world):
    # Same chunks, generated in different request shapes/order.
    a = make_world()
    a.generate(rect=(0, 0, 4, 4))

    b = make_world()
    b.generate(coords={ChunkCoord(3, 1)})
    b.generate(coords={ChunkCoord(0, 0), ChunkCoord(2, 3)})
    b.generate(rect=(0, 0, 3, 4))
    b.generate(rect=(3, 0, 4, 4))

    for coord in COORDS:
        assert (
            a.get_chunk(coord).cells == b.get_chunk(coord).cells
        ), coord


def test_rng_streams_are_fixed():
    x = RNG(12345)
    y = RNG(12345)
    seq_a = [x.next_float() for _ in range(16)]
    seq_b = [y.next_float() for _ in range(16)]
    assert seq_a == seq_b
    assert len(set(seq_a)) == 16


def test_noise_is_coordinate_continuous():
    # Adjacent cells differ far less than distant cells on average.
    import statistics

    near = [abs(fbm2(9, x, 50) - fbm2(9, x + 1, 50)) for x in range(300)]
    far = [abs(fbm2(9, x, 50) - fbm2(9, x + 64, 50)) for x in range(300)]
    assert statistics.mean(near) < statistics.mean(far) / 3
    for x in range(0, 200, 13):
        v = value_noise2(9, x, -x)
        assert 0.0 <= v < 1.0


def _stability_script(seed: int, rounds: int) -> str:
    return f"""
from worldgen import World, WorldConfig, ChunkCoord
import hashlib
w = World(WorldConfig(seed={seed}, max_propagation_rounds={rounds}))
w.generate(rect=(0, 0, 4, 4))
h = hashlib.sha256()
for x in range(4):
    for z in range(4):
        for cell in w.get_chunk(ChunkCoord(x, z)).cells:
            h.update(repr(cell).encode())
print(h.hexdigest())
"""


def test_stable_across_interpreter_runs(world, config):
    """The in-process digest must match a fresh interpreter exactly."""
    world.generate(rect=RECT)
    expected = _digest_world(world)
    repo = Path(__file__).resolve().parents[1]
    out = subprocess.run(
        [sys.executable, "-c", _stability_script(config.seed,
                                                 config.max_propagation_rounds)],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == expected


@pytest.mark.parametrize("seed", [0, 1, 42, -7, 2**40])
def test_many_seeds_all_reproducible(seed):
    from worldgen import World, WorldConfig

    cfg = WorldConfig(seed=seed)
    a = World(cfg)
    b = World(WorldConfig(seed=seed))
    a.generate(rect=(0, 0, 2, 2))
    b.generate(rect=(0, 0, 2, 2))
    coords = [ChunkCoord(x, z) for x in range(2) for z in range(2)]
    for c in coords:
        assert a.get_chunk(c).cells == b.get_chunk(c).cells
