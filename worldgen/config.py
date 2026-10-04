"""Configuration objects for worlds and regions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RegionParams:
    """Regional generation parameters.

    Biases are added to the global continuous base fields and therefore do
    not create hard seams: subsequent cross-cell relaxation blends them.
    """

    temperature_bias: float = 0.0
    moisture_bias: float = 0.0
    water_bias: float = 0.0
    elevation_bias: float = 0.0
    # Width (in world cells) of the smooth blend band at region borders.
    # The bias is multiplied by a smoothstep ramp of the distance to the
    # nearest region edge, so a parameter change never forms a hard seam.
    # Purely world-coordinate based => identical on both sides.
    border_blend_cells: int = 8


@dataclass(frozen=True, slots=True)
class WorldConfig:
    """Process-wide generation configuration.

    All fields that influence results participate in the deterministic
    parameter digest, so changing them invalidates cached output.
    """

    seed: int = 1337
    max_propagation_rounds: int = 4
    convergence_tolerance: float = 0.0
    non_converged_policy: str = "accept"  # "accept" | "raise"
    lock_timeout_seconds: float = 30.0
    max_commit_retries: int = 16
    generation_timeout_seconds: float | None = None
