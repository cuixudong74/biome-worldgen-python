"""Immutable cell and chunk data model plus deterministic base generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from .config import RegionParams, WorldGenConfig
from .noise import fbm


@dataclass(frozen=True)
class Cell:
    """A single world cell.  Instances are immutable."""

    height: float
    temperature: float
    moisture: float
    vegetation: float
    base_moisture: float = 0.0
    base_temperature: float = 0.0


@dataclass(frozen=True)
class Chunk:
    """A square chunk of ``size * size`` cells.

    Chunks are immutable value objects: every update produces a new
    instance, so a reader always observes a complete version and a commit
    can never tear a chunk across threads.
    """

    cx: int
    cz: int
    size: int
    cells: tuple[Cell, ...]
    params: RegionParams
    version: int = 1

    def index(self, lx: int, lz: int) -> int:
        return lz * self.size + lx

    def cell(self, lx: int, lz: int) -> Cell:
        return self.cells[self.index(lx, lz)]

    def with_cells(self, cells: tuple[Cell, ...], version: int | None = None) -> "Chunk":
        return Chunk(
            cx=self.cx,
            cz=self.cz,
            size=self.size,
            cells=cells,
            params=self.params,
            version=self.version + 1 if version is None else version,
        )

    def __iter__(self) -> Iterator[Cell]:
        return iter(self.cells)


def _round(value: float, decimals: int) -> float:
    return round(value, decimals)


def generate_base_chunk(
    seed: int,
    config: WorldGenConfig,
    params: RegionParams,
    cx: int,
    cz: int,
) -> Chunk:
    """Generate the *base* (pre-propagation) chunk.

    Terrain height is a pure function of the seed and world coordinates;
    temperature and moisture start from noise plus region offsets;
    vegetation is seeded at zero and grown during propagation.
    """
    size = config.chunk_size
    cells: list[Cell] = []
    for lz in range(size):
        for lx in range(size):
            wx = cx * size + lx
            wz = cz * size + lz
            height = fbm(seed, wx * 0.05, wz * 0.05, salt=0, octaves=config.terrain_octaves)
            height = _round(height, config.field_decimals)

            temp_noise = fbm(seed, wx * 0.08 + 1000.0, wz * 0.08, salt=1,
                              octaves=config.climate_octaves)
            temperature = (
                config.base_temperature
                - config.latitude_factor * abs(wz)
                - config.temperature_lapse * max(0.0, height - config.sea_level)
                + (temp_noise - 0.5) * 6.0
                + params.temperature_offset
            )

            moist_noise = fbm(seed, wx * 0.07, wz * 0.07 + 1000.0, salt=2,
                              octaves=config.climate_octaves)
            moisture = moist_noise * 0.5 + 0.35 + params.moisture_offset
            if height < config.sea_level:
                moisture = 1.0

            base_temperature = temperature
            cells.append(Cell(
                height=height,
                temperature=_round(temperature, config.field_decimals),
                moisture=_round(min(1.0, max(0.0, moisture)), config.field_decimals),
                vegetation=0.0,
                base_moisture=1.0 if height < config.sea_level else _round(
                    min(1.0, max(0.0, moisture)), config.field_decimals
                ),
                base_temperature=_round(base_temperature, config.field_decimals),
            ))
    return Chunk(cx=cx, cz=cz, size=size, cells=tuple(cells), params=params, version=1)
