"""Configuration objects for the world generator.

All configuration is immutable (frozen dataclasses) so that a world, once
created, always generates with the same parameters.  See ``docs/config.md``
for the meaning of every field.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegionParams:
    """Per-region (and per-chunk) generation parameters.

    These parameters only affect the *surface* fields (temperature,
    moisture, vegetation potential).  Terrain height is derived solely from
    the world seed so that it can never diverge between neighbours.
    """

    temperature_offset: float = 0.0
    """Extra temperature added to every cell of the region (degrees C)."""

    moisture_offset: float = 0.0
    """Extra base moisture added to every cell of the region (0..1)."""

    fertility: float = 1.0
    """Vegetation growth multiplier; 0 suppresses vegetation entirely."""

    seed_salt: int = 0
    """Extra seed mixed into the noise; allows distinct regions with
    otherwise identical parameters to differ deterministically."""


@dataclass(frozen=True)
class WorldGenConfig:
    """Tunable generation parameters.  See ``docs/config.md``."""

    # ---- chunk / grid -------------------------------------------------
    chunk_size: int = 16
    """Edge length of a chunk in cells (chunks are square)."""

    # ---- terrain ------------------------------------------------------
    sea_level: float = 0.45
    """Cells with height below this are ocean (0..1)."""

    terrain_octaves: int = 4
    """fBm octaves used for terrain height."""

    # ---- climate ------------------------------------------------------
    base_temperature: float = 26.0
    """Mean equatorial temperature (degrees C)."""

    latitude_factor: float = 0.07
    """Temperature decrease per block of z-distance from the origin."""

    temperature_lapse: float = 7.5
    """Temperature decrease per unit of terrain height (deg C)."""

    climate_octaves: int = 3
    """fBm octaves used for the temperature / moisture noise fields."""

    # ---- propagation --------------------------------------------------
    propagation_enabled: bool = True
    """When False, chunks are generated without cross-chunk propagation
    (useful for tests / debugging)."""

    max_propagation_rounds: int = 300
    """Hard cap on propagation rounds.  The fixed point is contractive so
    this is only a safety net; generation never deadlocks."""

    convergence_epsilon: float = 1e-12
    """Iteration stops when the maximum per-cell field change of a round
    drops below this value."""

    field_decimals: int = 6
    """Cell fields are rounded to this many decimals at commit time.  This
    makes far-field perturbation deterministically identical after an
    incremental change and immune to float summation order."""

    moisture_diffusion: float = 0.20
    """Fraction of moisture equalised with existing neighbours per round."""

    moisture_advection: float = 0.30
    """Fraction of moisture replaced by the up-wind neighbour per round."""

    wind_dx: int = 1
    """Wind direction x component (prevailing wind blows towards +x)."""

    wind_dz: int = 0
    """Wind direction z component."""

    ocean_evaporation: float = 0.15
    """Rate at which ocean cells push moisture towards 1.0 per round."""

    rain_shadow: float = 0.12
    """Rate at which elevated land sheds moisture (orographic precipitation)."""

    groundwater_recharge: float = 0.30
    """Rate at which land moisture relaxes back to its noise-derived base
    value (groundwater), keeping continents from drying out completely."""

    temperature_diffusion: float = 0.04
    """Fraction of temperature equalised with neighbours per round."""

    temperature_equilibration: float = 0.35
    """Rate at which temperature relaxes back to its seed-derived base
    (latitude + altitude + climate noise), gapping the long-wave diffusion
    mode so propagation converges quickly."""

    vegetation_spread: float = 0.10
    """Fraction of vegetation equalised with neighbours per round."""

    vegetation_growth: float = 0.40
    """Rate at which vegetation grows towards the local climate potential."""

    # ---- runtime ------------------------------------------------------
    worker_threads: int = 4
    """Worker threads used for parallel chunk base generation."""

    default_timeout: float = 60.0
    """Default operation timeout in seconds."""
