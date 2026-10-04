"""Region parameter registry with deterministic resolution and digest.

Regions are axis-aligned chunk-grid rectangles. Where they overlap the
*region id* breaks the tie (lexicographically greatest id wins); that rule
is independent of insertion order, which keeps concurrent/retried requests
identical. The digest folds in the seed too, so a chunk records exactly
which parameter state produced it.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass

from worldgen.config import RegionParams
from worldgen.coords import CHUNK_SIZE, ChunkCoord, chunks_in_rect
from worldgen.terrain import ParamResolver


@dataclass(frozen=True, slots=True)
class Region:
    """An axis-aligned (chunk-grid) rectangle with its own parameters."""

    region_id: str
    cx0: int
    cz0: int
    cx1: int
    cz1: int
    params: RegionParams

    def __post_init__(self) -> None:
        if self.cx1 <= self.cx0 or self.cz1 <= self.cz0:
            raise ValueError("region rectangle must be non-empty")

    def contains_cell(self, wx: int, wz: int) -> bool:
        x0, z0 = self.cx0 * CHUNK_SIZE, self.cz0 * CHUNK_SIZE
        x1, z1 = self.cx1 * CHUNK_SIZE, self.cz1 * CHUNK_SIZE
        return x0 <= wx < x1 and z0 <= wz < z1

    def contains_chunk(self, coord: ChunkCoord) -> bool:
        return (
            self.cx0 <= coord.cx < self.cx1
            and self.cz0 <= coord.cz < self.cz1
        )

    def chunks(self) -> tuple[ChunkCoord, ...]:
        return chunks_in_rect(self.cx0, self.cz0, self.cx1, self.cz1)


class ParamRegistry(ParamResolver):
    """Latest-wins overlapping region registry (stable id tie-break)."""

    DEFAULT_PARAMS = RegionParams()

    def __init__(self, seed: int = 0) -> None:
        self._lock = threading.RLock()
        self._seed = seed
        self._regions: dict[str, Region] = {}
        self._digest_cache: str | None = None

    def _invalidate(self) -> None:
        self._digest_cache = None

    def upsert_region(self, region: Region) -> set[ChunkCoord]:
        """Insert/replace a region; return chunks intersecting it."""
        with self._lock:
            self._regions[region.region_id] = region
            self._invalidate()
            return set(region.chunks())

    def remove_region(self, region_id: str) -> set[ChunkCoord]:
        with self._lock:
            region = self._regions.pop(region_id, None)
            self._invalidate()
            return set(region.chunks()) if region is not None else set()

    def params_for_cell(self, wx: int, wz: int) -> RegionParams:
        # Snapshot under the lock so the winning region cannot change while
        # a pass resolves its biases.
        with self._lock:
            candidates = tuple(self._regions.values())
        winner_id = ""
        winner = None
        for region in candidates:
            if region.contains_cell(wx, wz):
                if winner is None or region.region_id > winner_id:
                    winner, winner_id = region, region.region_id
        return winner.params if winner is not None else self.DEFAULT_PARAMS

    def region_bounds_for_cell(
        self, wx: int, wz: int
    ) -> tuple[int, int, int, int] | None:
        with self._lock:
            candidates = tuple(self._regions.values())
        winner_id = ""
        winner = None
        for region in candidates:
            if region.contains_cell(wx, wz):
                if winner is None or region.region_id > winner_id:
                    winner, winner_id = region, region.region_id
        if winner is None:
            return None
        return (
            winner.cx0 * CHUNK_SIZE,
            winner.cz0 * CHUNK_SIZE,
            winner.cx1 * CHUNK_SIZE,
            winner.cz1 * CHUNK_SIZE,
        )

    def digest(self) -> str:
        """Stable digest of the seed and full parameter state."""
        with self._lock:
            if self._digest_cache is not None:
                return self._digest_cache
            h = hashlib.sha256()
            h.update(b"params-v1|")
            h.update(str(self._seed).encode())
            h.update(b"|regions=")
            for region_id in sorted(self._regions):
                r = self._regions[region_id]
                p = r.params
                h.update(
                    f"{r.region_id};{r.cx0},{r.cz0},{r.cx1},{r.cz1};"
                    f"{p.temperature_bias!r},{p.moisture_bias!r},"
                    f"{p.water_bias!r},{p.elevation_bias!r},"
                    f"{p.border_blend_cells}\n".encode()
                )
            self._digest_cache = h.hexdigest()
            return self._digest_cache

    def state_snapshot(self) -> tuple[Region, ...]:
        with self._lock:
            return tuple(self._regions.values())

    def restore(self, regions) -> None:
        """Replace the whole registry state (used by snapshot rollback)."""
        with self._lock:
            self._regions = {r.region_id: r for r in regions}
            self._invalidate()
