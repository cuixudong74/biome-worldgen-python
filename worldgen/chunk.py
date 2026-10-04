"""Immutable chunk and cell value objects."""

from __future__ import annotations

from dataclasses import dataclass

from worldgen.coords import CHUNK_SIZE, ChunkCoord
from worldgen.types import Biome

# A neutral placeholder cell used to size empty grids; never committed.
_PLACEHOLDER = None


@dataclass(frozen=True, slots=True)
class ChunkCell:
    elevation: float
    temperature: float
    moisture: float
    water: float
    biome: Biome


@dataclass(frozen=True, slots=True)
class Chunk:
    """An immutable, versioned chunk.

    ``cells`` is a flat row-major tuple of length ``CHUNK_SIZE ** 2``.
    Readers always observe a whole chunk version, never a half update:
    chunks are built fully before being installed in the store.
    """

    coord: ChunkCoord
    version: int
    params_digest: str
    cells: tuple[ChunkCell, ...]

    def cell_at(self, lx: int, lz: int) -> ChunkCell:
        if not (0 <= lx < CHUNK_SIZE and 0 <= lz < CHUNK_SIZE):
            raise IndexError(f"local cell out of range: {(lx, lz)}")
        return self.cells[lz * CHUNK_SIZE + lx]


def empty_cells() -> tuple[ChunkCell, ...]:
    """Return a correctly sized tuple of None (placeholder only)."""
    return (None,) * (CHUNK_SIZE * CHUNK_SIZE)
