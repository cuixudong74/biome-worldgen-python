"""Chunk / world coordinate helpers.

The world is an infinite grid of square chunks. Chunk ``(cx, cz)`` owns the
world cells ``wx`` in ``[cx*CHUNK_SIZE, (cx+1)*CHUNK_SIZE)`` and likewise
for ``wz``. Negative chunk coordinates are supported and use floor
division, so the mapping never double-owns or omits a cell.
"""

from __future__ import annotations

from dataclasses import dataclass

CHUNK_SIZE: int = 16


def floor_div(a: int, b: int) -> int:
    """Floor division (Python ``//`` already floors; kept for clarity)."""
    return a // b


@dataclass(frozen=True, slots=True, order=True)
class ChunkCoord:
    """Coordinate of a chunk in chunk-grid space."""

    cx: int
    cz: int

    @property
    def key(self) -> tuple[int, int]:
        return (self.cx, self.cz)

    def neighbors(self) -> tuple["ChunkCoord", ...]:
        return (
            ChunkCoord(self.cx + 1, self.cz),
            ChunkCoord(self.cx - 1, self.cz),
            ChunkCoord(self.cx, self.cz + 1),
            ChunkCoord(self.cx, self.cz - 1),
        )

    def world_origin(self) -> tuple[int, int]:
        """World-space coordinate of the chunk's minimum corner."""
        return (self.cx * CHUNK_SIZE, self.cz * CHUNK_SIZE)


def cell_to_chunk(wx: int, wz: int) -> ChunkCoord:
    return ChunkCoord(floor_div(wx, CHUNK_SIZE), floor_div(wz, CHUNK_SIZE))


def chunks_in_rect(
    cx0: int, cz0: int, cx1: int, cz1: int
) -> tuple[ChunkCoord, ...]:
    """All chunk coords with ``cx0 <= cx < cx1`` and likewise for ``cz``."""
    if cx1 <= cx0 or cz1 <= cz0:
        return ()
    return tuple(
        ChunkCoord(cx, cz)
        for cx in range(cx0, cx1)
        for cz in range(cz0, cz1)
    )


def rect_expand(
    rect: tuple[int, int, int, int], radius: int
) -> tuple[int, int, int, int]:
    cx0, cz0, cx1, cz1 = rect
    return (cx0 - radius, cz0 - radius, cx1 + radius, cz1 + radius)
