"""Deterministic, stateless noise.

Every value is a pure function of ``(seed, x, z)`` so generation never
depends on global random state or on the order in which chunks are
produced.  This is the foundation of reproducibility and of order
independent boundary continuity.
"""

from __future__ import annotations

MASK64 = (1 << 64) - 1


def _mix64(x: int) -> int:
    """splitmix64 finaliser; maps 64 bits to 64 bits with good avalanche."""
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
    return x ^ (x >> 31)


def hash2(seed: int, x: int, z: int, salt: int = 0) -> int:
    """Return a deterministic 64-bit hash for the integer cell ``(x, z)``."""
    h = _mix64(seed & MASK64)
    h = _mix64((h ^ _mix64(x & MASK64)) & MASK64)
    h = _mix64((h ^ _mix64(z & MASK64)) & MASK64)
    h = _mix64((h ^ _mix64(salt & MASK64)) & MASK64)
    return h


def hash01(seed: int, x: int, z: int, salt: int = 0) -> float:
    """Return a deterministic pseudo-random float in ``[0, 1)``."""
    return (hash2(seed, x, z, salt) >> 11) / float(1 << 53)


def _smooth(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def value_noise(seed: int, x: float, z: float, salt: int = 0) -> float:
    """Bilinear value noise in ``[0, 1]`` at continuous coordinates."""
    x0 = int(x) if x >= 0 else int(x) - 1
    z0 = int(z) if z >= 0 else int(z) - 1
    fx = x - x0
    fz = z - z0
    v00 = hash01(seed, x0, z0, salt)
    v10 = hash01(seed, x0 + 1, z0, salt)
    v01 = hash01(seed, x0, z0 + 1, salt)
    v11 = hash01(seed, x0 + 1, z0 + 1, salt)
    sx = _smooth(fx)
    sz = _smooth(fz)
    a = v00 + (v10 - v00) * sx
    b = v01 + (v11 - v01) * sx
    return a + (b - a) * sz


def fbm(
    seed: int,
    x: float,
    z: float,
    salt: int = 0,
    octaves: int = 4,
    lacunarity: float = 2.0,
    gain: float = 0.5,
) -> float:
    """Fractional Brownian motion; returns a value roughly in ``[0, 1]``."""
    total = 0.0
    amplitude = 1.0
    frequency = 1.0
    norm = 0.0
    for _ in range(octaves):
        total += amplitude * value_noise(seed, x * frequency, z * frequency, salt)
        norm += amplitude
        amplitude *= gain
        frequency *= lacunarity
    return total / norm
