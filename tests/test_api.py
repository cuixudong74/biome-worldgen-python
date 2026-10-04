"""Smoke tests for package-level ergonomics and documented defaults."""

from __future__ import annotations

import pytest

from worldgen import (
    Biome,
    Chunk,
    ChunkCell,
    ChunkCoord,
    SnapshotState,
    World,
)
from worldgen.coords import CHUNK_SIZE, cell_to_chunk


def test_package_exports():
    assert CHUNK_SIZE == 16
    assert Biome.FOREST.value == "forest"
    assert SnapshotState.COMMITTED.value == "committed"


def test_chunk_coord_helpers():
    assert cell_to_chunk(15, 15) == ChunkCoord(0, 0)
    assert cell_to_chunk(16, 0) == ChunkCoord(1, 0)
    assert cell_to_chunk(-1, -1) == ChunkCoord(-1, -1)
    assert ChunkCoord(2, -3).world_origin() == (32, -48)
    assert len(ChunkCoord(0, 0).neighbors()) == 4


def test_default_world_generates_grid():
    world = World()  # all defaults
    chunks = world.generate(rect=(0, 0, 2, 2))
    assert len(chunks) == 4
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(len(c.cells) == CHUNK_SIZE * CHUNK_SIZE for c in chunks)
    assert all(isinstance(cell, ChunkCell) for c in chunks for cell in c.cells)
    assert all(cell.biome in Biome for c in chunks for cell in c.cells)


def test_invalid_arguments():
    world = World()
    with pytest.raises(ValueError):
        world.generate()
    with pytest.raises(ValueError):
        world.generate(rect=(1, 1, 1, 1))  # empty rect
