"""Deterministic base terrain/climate field synthesis.

All functions are pure in ``(seed, world coordinate)``: there is no chunk
local state, so two chunks meeting at a seam derive the exact same value
for a shared world cell. Lattice value noise (smoothstep interpolation)
additionally makes neighbours close in value, so cross-seam gradients are
continuous rather than just equal at duplicated cells.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from worldgen.config import RegionParams
from worldgen.fields import (
    ELEVATION,
    Fields,
    MOISTURE,
    TEMPERATURE,
    WATER,
)
from worldgen.rng import RNG, mix64

# Feature seeds (fixed component mixing constants for the world seed).
_SEED_ELEVATION = 0xE1E_0001
_SEED_TEMPERATURE = 0x7E47_0002
_SEED_MOISTURE = 0x4015_0003
_SEED_WATER = 0x4A7E_0004


@runtime_checkable
class ParamResolver(Protocol):
    def params_for_cell(self, wx: int, wz: int) -> RegionParams: ...

    def region_bounds_for_cell(
        self, wx: int, wz: int
    ) -> tuple[int, int, int, int] | None:
        """Cell bounds (inclusive/exclusive world coords) of the winning
        region, or ``None`` for default params."""
        ...


def _hash2(seed: int, x: int, z: int) -> int:
    """Deterministic 64-bit hash of a lattice point."""
    return mix64(mix64(seed, x & 0xFFFFFFFFFFFFFFFF), z)


def _smooth(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def value_noise2(seed: int, x: int, z: int) -> float:
    """World-continuous lattice value noise in ``[0, 1)``."""
    x0, z0 = x // 16, z // 16
    fx, fz = (x - x0 * 16) / 16.0, (z - z0 * 16) / 16.0
    sx, sz = _smooth(fx), _smooth(fz)

    def h(dx: int, dz: int) -> float:
        return (_hash2(seed, x0 + dx, z0 + dz) >> 11) / float(1 << 53)

    v00, v10 = h(0, 0), h(1, 0)
    v01, v11 = h(0, 1), h(1, 1)
    top = v00 + (v10 - v00) * sx
    bot = v01 + (v11 - v01) * sx
    return top + (bot - top) * sz


def fbm2(seed: int, x: int, z: int, octaves: int = 4) -> float:
    """Fractal value noise in approximately ``[0, 1)``.

    The value is independent of which chunk or pass requests it.
    """
    total = 0.0
    amplitude = 0.5
    norm = 0.0
    frequency = 1
    for octave in range(octaves):
        total += amplitude * value_noise2(
            seed + octave * 0x9E3779B1, x * frequency, z * frequency
        )
        norm += amplitude
        amplitude *= 0.5
        frequency *= 2
    return total / norm


def base_elevation(seed: int, wx: int, wz: int) -> float:
    continents = fbm2(seed ^ _SEED_ELEVATION, wx, wz, octaves=4)
    hills = 0.25 * fbm2(seed ^ _SEED_ELEVATION ^ 0x111, wx * 4, wz * 4, 3)
    return continents + hills


def base_temperature(seed: int, wx: int, wz: int) -> float:
    # Equator warm, poles cool; latitude is wz at world scale.
    latitude = wz / 128.0
    zonal = 0.5 - 0.42 * min(1.0, abs(latitude))
    noise = 0.25 * (fbm2(seed ^ _SEED_TEMPERATURE, wx, wz, 3) - 0.5)
    return 0.5 + zonal + noise


def base_moisture(seed: int, wx: int, wz: int) -> float:
    return fbm2(seed ^ _SEED_MOISTURE, wx, wz, octaves=4)


def base_water(seed: int, wx: int, wz: int) -> float:
    v = fbm2(seed ^ _SEED_WATER, wx, wz, octaves=3)
    return max(0.0, v - 0.55) / 0.45


def base_field(arr_id: int, seed: int, wx: int, wz: int) -> float:
    if arr_id == ELEVATION:
        return base_elevation(seed, wx, wz)
    if arr_id == TEMPERATURE:
        return base_temperature(seed, wx, wz)
    if arr_id == MOISTURE:
        return base_moisture(seed, wx, wz)
    if arr_id == WATER:
        return base_water(seed, wx, wz)
    raise ValueError(f"unknown field id {arr_id}")


def _ramp(d: float, width: float) -> float:
    if width <= 0.0:
        return 1.0
    if d >= width:
        return 1.0
    t = d / width
    return t * t * (3.0 - 2.0 * t)


def effective_params(
    resolver: ParamResolver, wx: int, wz: int
) -> RegionParams:
    """Resolve params with the world-continuous border blend applied."""
    p = resolver.params_for_cell(wx, wz)
    width = float(getattr(p, "border_blend_cells", 0) or 0)
    if width <= 0.0:
        return p
    bounds = resolver.region_bounds_for_cell(wx, wz)
    if bounds is None:
        return p
    x0, z0, x1, z1 = bounds
    d = float(min(wx - x0, x1 - 1 - wx, wz - z0, z1 - 1 - wz))
    k = _ramp(d + 0.5, width)
    return RegionParams(
        temperature_bias=p.temperature_bias * k,
        moisture_bias=p.moisture_bias * k,
        water_bias=p.water_bias * k,
        elevation_bias=p.elevation_bias * k,
        border_blend_cells=p.border_blend_cells,
    )


def initialize_fields(
    fields: Fields, seed: int, resolver: ParamResolver, rng: RNG
) -> None:
    """Fill every mutable cell with biased base fields (in place).

    ``rng`` is accepted for API symmetry/future use; current base fields are
    coordinate-hash based and do not draw from it.
    """
    del rng  # deterministic base synthesis needs no stream state
    size = fields.width * fields.height
    fields.elevation = [0.0] * size
    fields.temperature = [0.0] * size
    fields.moisture = [0.0] * size
    fields.water = [0.0] * size
    for wz in range(fields.wz0, fields.wz1):
        for wx in range(fields.wx0, fields.wx1):
            p = effective_params(resolver, wx, wz)
            i = fields._index(wx, wz)
            fields.elevation[i] = base_elevation(seed, wx, wz) + p.elevation_bias
            fields.temperature[i] = (
                base_temperature(seed, wx, wz) + p.temperature_bias
            )
            fields.moisture[i] = base_moisture(seed, wx, wz) + p.moisture_bias
            fields.water[i] = base_water(seed, wx, wz) + p.water_bias
