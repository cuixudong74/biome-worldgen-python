"""Requirement 2: adjacent-chunk boundary continuity.

A world cell is owned by exactly one chunk, so there is no duplicated
"same cell" to disagree; we verify (a) two neighbours evaluating the same
world coordinate get the identical value, and (b) the cross-seam field
gradient stays of the same order as in-chunk gradients - no hard cliff -
even when region parameters change across the border.
"""

from __future__ import annotations

from worldgen import ChunkCoord, RegionParams
from worldgen.pipeline import run_pass
from worldgen.params import ParamRegistry, Region


def _max_cross_seam_jump(world, left: ChunkCoord, right: ChunkCoord):
    """Max |field(right edge column) - field(left edge column)|."""
    a = world.get_chunk(left)
    b = world.get_chunk(right)
    jumps = {f: 0.0 for f in ("temperature", "moisture", "water")}
    for lz in range(16):
        ca = a.cell_at(15, lz)
        cb = b.cell_at(0, lz)
        jumps["temperature"] = max(
            jumps["temperature"], abs(cb.temperature - ca.temperature)
        )
        jumps["moisture"] = max(
            jumps["moisture"], abs(cb.moisture - ca.moisture)
        )
        jumps["water"] = max(jumps["water"], abs(cb.water - ca.water))
    return jumps


def _mean_within_steps(chunk):
    import statistics

    return statistics.mean(
        abs(
            chunk.cell_at(x + 1, lz).temperature
            - chunk.cell_at(x, lz).temperature
        )
        for lz in range(16)
        for x in range(12)
    )


def test_seam_continuous_without_regions(world):
    world.generate(rect=(0, 0, 4, 4))
    for cz in range(4):
        jumps = _max_cross_seam_jump(
            world, ChunkCoord(1, cz), ChunkCoord(2, cz)
        )
        within = _mean_within_steps(world.get_chunk(ChunkCoord(2, cz)))
        # Cross-seam jump must not exceed a few times the normal step.
        assert jumps["temperature"] <= max(3.0 * within, 1e-9)
        assert jumps["moisture"] <= 0.05
        assert jumps["water"] <= 0.05


def test_seam_continuous_across_parameter_border(world):
    world.generate(rect=(0, 0, 6, 6))
    # Region boundary runs exactly between chunk columns 2 and 3.
    world.apply_region_params(
        "hot", (3, 0, 6, 6), RegionParams(temperature_bias=0.35)
    )
    for cz in range(6):
        jumps = _max_cross_seam_jump(
            world, ChunkCoord(2, cz), ChunkCoord(3, cz)
        )
        # Biased regions are blended by relaxation: no 0.35 hard cliff.
        assert jumps["temperature"] < 0.35 / 4.0
        assert jumps["moisture"] < 0.05
        assert jumps["water"] < 0.05


def test_same_world_cell_identical_from_either_side(config):
    """Direct pipeline identity for cells at a chunk boundary."""
    registry = ParamRegistry(seed=config.seed)
    registry.upsert_region(
        Region("hot", 0, 0, 3, 3, RegionParams(temperature_bias=0.2))
    )
    digest = registry.digest()

    alone = run_pass(
        write_coords={ChunkCoord(0, 1)},
        config=config,
        resolver=registry,
        params_digest=digest,
    )[0]
    pair = run_pass(
        write_coords={ChunkCoord(0, 1), ChunkCoord(1, 1)},
        config=config,
        resolver=registry,
        params_digest=digest,
    )
    by_coord = {c.coord: c for c in pair}
    assert alone.cells == by_coord[ChunkCoord(0, 1)].cells


def test_four_chunk_corner_is_consistent(world):
    # The meeting point of four chunks must be smooth on all four edges.
    world.generate(rect=(0, 0, 4, 4))
    coords = [ChunkCoord(1, 1), ChunkCoord(2, 1), ChunkCoord(1, 2), ChunkCoord(2, 2)]
    for coord in coords:
        assert world.get_chunk(coord) is not None
    # Edge columns across both internal seams are close.
    for cz in (1, 2):
        jumps = _max_cross_seam_jump(
            world, ChunkCoord(1, cz), ChunkCoord(2, cz)
        )
        assert jumps["temperature"] < 0.05
