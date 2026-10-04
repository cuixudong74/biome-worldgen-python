"""Deterministic, platform-independent pseudo-random primitives.

We only use fixed-width integer arithmetic (SplitMix64), so streams are
identical on every Python build/platform. This is the foundation of the
"same seed + same params => identical cells" guarantee.
"""

from __future__ import annotations

MASK64 = (1 << 64) - 1


def _rotl(x: int, k: int) -> int:
    return ((x << k) | (x >> (64 - k))) & MASK64


def splitmix64(state: int) -> tuple[int, int]:
    """One SplitMix64 step. Returns ``(new_state, value)``."""
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    z = z ^ (z >> 31)
    return state, z & MASK64


def mix64(a: int, b: int = 0) -> int:
    """Deterministically mix two 64-bit integers into one."""
    _, z = splitmix64((a & MASK64) ^ splitmix64(b & MASK64)[1])
    return z


class RNG:
    """Small deterministic PRNG used for per-cell jitter.

    The stream only depends on ``seed``; call order is fixed by the
    generation code so concurrent requests draw identical numbers.
    """

    __slots__ = ("_state",)

    def __init__(self, seed: int) -> None:
        self._state = seed & MASK64

    def next_u64(self) -> int:
        self._state, value = splitmix64(self._state)
        return value

    def next_float(self) -> float:
        """Uniform float in ``[0.0, 1.0)`` with a fixed 53-bit draw."""
        return (self.next_u64() >> 11) / float(1 << 53)
