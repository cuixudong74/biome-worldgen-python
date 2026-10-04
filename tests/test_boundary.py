"""Boundary continuity across chunk seams.

Climate, moisture and vegetation cross seams during propagation, so the
same world generated via different chunk orders must produce identical
cells everywhere -- especially adjacent to seams.
"""

from __future__ import annotations

from biome_worldgen import World, WorldGenConfig


def _cfg(**kw) -> WorldGenConfig:
    base = dict(chunk_size=4, max_propagation_rounds=300, convergence_epsilon=1e-12)
    base.update(kw)
    return WorldGenConfig(**base)


def _cells(world: World):
    out = {}
    for key in sorted(world.chunk_keys()):
        chunk = world.get_chunk(*key)
        for i, cell in enumerate(chunk.cells):
            out[(key, i)] = cell
    return out


def test_batch_vs_column_order():
    w1 = World(seed=101, config=_cfg())
    w2 = World(seed=101, config=_cfg())
    w1.generate_region(0, 0, 4, 4)
    for cx in range(5):
        w2.generate_region(cx, 0, cx, 4)
    c1 = _cells(w1)
    c2 = _cells(w2)
    assert set(c1) == set(c2)
    mismatches = [k for k in c1 if c1[k] != c2[k]]
    assert not mismatches, f"{len(mismatches)} seam/order mismatches"


def test_batch_vs_row_order():
    w1 = World(seed=202, config=_cfg())
    w2 = World(seed=202, config=_cfg())
    w1.generate_region(0, 0, 4, 4)
    for cz in range(5):
        w2.generate_region(0, cz, 4, cz)
    c1 = _cells(w1)
    c2 = _cells(w2)
    mismatches = [k for k in c1 if c1[k] != c2[k]]
    assert not mismatches, f"{len(mismatches)} row-order mismatches"


def test_independent_adjacent_chunks_agree_at_seam():
    """Generate two adjacent chunks independently (each with its own
    propagation over the growing set), then verify the seam cells match
    a batch-generated reference exactly."""
    ref = World(seed=303, config=_cfg())
    ref.generate_region(0, 0, 1, 1)

    w = World(seed=303, config=_cfg())
    w.generate_region(0, 0, 0, 1)  # left column
    w.generate_region(1, 0, 1, 1)  # right column (triggers cross-seam prop)

    for key in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        a = ref.get_chunk(*key)
        b = w.get_chunk(*key)
        assert a.cells == b.cells, f"chunk {key} seam mismatch"


def test_seam_gradient_is_continuous():
    """Across a vertical seam, adjacent cells should not jump
    discontinuously: the cross-seam difference must be comparable to
    within-chunk neighbour differences (no cliff at the seam)."""
    w = World(seed=404, config=_cfg())
    w.generate_region(0, 0, 3, 3)
    size = w.config.chunk_size
    seam_diffs = []
    inner_diffs = []
    for cz in range(4):
        for cx in range(3):
            left = w.get_chunk(cx, cz)
            right = w.get_chunk(cx + 1, cz)
            for lz in range(size):
                a = left.cell(size - 1, lz)
                b = right.cell(0, lz)
                seam_diffs.append(abs(a.moisture - b.moisture))
                seam_diffs.append(abs(a.vegetation - b.vegetation))
                # inner neighbour pair for comparison
                c = left.cell(size - 2, lz)
                inner_diffs.append(abs(c.moisture - a.moisture))
                inner_diffs.append(abs(c.vegetation - a.vegetation))
    seam_avg = sum(seam_diffs) / len(seam_diffs)
    inner_avg = sum(inner_diffs) / len(inner_diffs)
    # Seam crossing should be no worse than an inner neighbour step.
    assert seam_avg <= inner_avg * 1.5 + 1e-6, (seam_avg, inner_avg)


def test_no_contradictory_cells():
    """Every cell has exactly one value (no duplicate/contradictory
    copies), and re-reading a chunk yields the same object."""
    w = World(seed=505, config=_cfg())
    w.generate_region(0, 0, 2, 2)
    for key in w.chunk_keys():
        a = w.get_chunk(*key)
        b = w.get_chunk(*key)
        assert a is b
        assert a.cells == b.cells
