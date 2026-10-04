"""Deterministic classification of final cell state into biomes."""

from __future__ import annotations

from worldgen.chunk import ChunkCell
from worldgen.coords import CHUNK_SIZE
from worldgen.fields import (
    ELEVATION,
    Fields,
    MOISTURE,
    TEMPERATURE,
    WATER,
)
from worldgen.rng import RNG, mix64
from worldgen.terrain import ParamResolver
from worldgen.types import Biome


def classify(
    elevation: float,
    temperature: float,
    moisture: float,
    water: float,
    jitter: float,
    params: object,
) -> Biome:
    """Pure, deterministic biome rule.

    Water level is evaluated first, then elevation, then the
    temperature/moisture climate quadrants. ``jitter`` in ``[0, 1)`` only
    nudches boundary comparisons by a fixed, hash-seeded amount, so the same
    cell always gets the same jitter regardless of pass/order.
    """
    j = (jitter - 0.5) * 0.02

    if water >= 0.55:
        return Biome.DEEP_OCEAN
    if water >= 0.18:
        return Biome.OCEAN

    if elevation >= 0.95 + j:
        return Biome.MOUNTAIN
    if elevation >= 0.82 + j:
        return Biome.SNOW if temperature < 0.45 else Biome.MOUNTAIN

    # Coastal flats next to open water.
    if water >= 0.10 and elevation < 0.30:
        return Biome.BEACH

    t = temperature + j
    m = moisture + j

    if t < 0.18:
        return Biome.TUNDRA if m < 0.45 else Biome.SNOW
    if t < 0.32:
        return Biome.TAIGA
    if t > 0.78 and m < 0.22:
        return Biome.DESERT
    if m < 0.30:
        return Biome.SAVANNA if t > 0.6 else Biome.GRASSLAND
    if m > 0.72 and t > 0.62:
        return Biome.RAINFOREST
    if m > 0.55:
        return Biome.FOREST
    return Biome.GRASSLAND


def build_chunk_cells(
    fields: Fields,
    cx: int,
    cz: int,
    resolver: ParamResolver,
    seed: int,
    rng: RNG,
) -> tuple[ChunkCell, ...]:
    """Extract the ``CHUNK_SIZE`` square cell tuple for ``(cx, cz)``.

    Iteration order is fixed (row-major) and jitter is derived from the
    world coordinate rather than from mutable RNG call order, keeping the
    result identical for concurrent or retried passes.
    """
    del rng
    ox, oz = cx * CHUNK_SIZE, cz * CHUNK_SIZE
    cells: list[ChunkCell] = []
    for lz in range(CHUNK_SIZE):
        for lx in range(CHUNK_SIZE):
            wx, wz = ox + lx, oz + lz
            elevation = fields.get(ELEVATION, wx, wz)
            temperature = fields.get(TEMPERATURE, wx, wz)
            moisture = fields.get(MOISTURE, wx, wz)
            water = fields.get(WATER, wx, wz)
            jitter = (
                mix64(mix64(seed & 0xFFFFFFFFFFFFFFFF, wx), wz) >> 11
            ) / float(1 << 53)
            biome = classify(
                elevation,
                temperature,
                moisture,
                water,
                jitter,
                resolver.params_for_cell(wx, wz),
            )
            cells.append(
                ChunkCell(
                    elevation=elevation,
                    temperature=temperature,
                    moisture=moisture,
                    water=water,
                    biome=biome,
                )
            )
    return tuple(cells)
