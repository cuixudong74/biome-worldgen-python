"""Dense rectangular scalar fields used during a generation pass.

A :class:`Fields` stores temperature/moisture/water over a rectangle of
*cells* in world coordinates. Only cells inside the rectangle are mutable;
everything outside is a read-only base value produced by a callback. That
boundary is why an incremental pass is bit-identical to a full rebuild as
long as the rectangle includes a halo as deep as the number of rounds: the
halo starts from, and the border always reads, the exact same base values
the global run would compute.

Arrays are row-major: index ``(wz - wz0) * width + (wx - wx0)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

#: Array ids, fixed order everywhere (deterministic arithmetic ordering).
ELEVATION = 0
TEMPERATURE = 1
MOISTURE = 2
WATER = 3

_ARRAYS = ("elevation", "temperature", "moisture", "water")

ExternalFn = Callable[[int, int, int], float]


@dataclass
class Fields:
    """Row-major float arrays over a world-space cell rectangle."""

    wx0: int
    wz0: int
    wx1: int
    wz1: int
    elevation: list[float]
    temperature: list[float]
    moisture: list[float]
    water: list[float]
    # external(array_id, wx, wz) -> base value for cells outside the rect.
    external: ExternalFn | None = None

    @property
    def width(self) -> int:
        return self.wx1 - self.wx0

    @property
    def height(self) -> int:
        return self.wz1 - self.wz0

    def _index(self, wx: int, wz: int) -> int:
        return (wz - self.wz0) * self.width + (wx - self.wx0)

    def contains(self, wx: int, wz: int) -> bool:
        return self.wx0 <= wx < self.wx1 and self.wz0 <= wz < self.wz1

    def _array(self, arr_id: int) -> list[float]:
        return getattr(self, _ARRAYS[arr_id])

    def get(self, arr_id: int, wx: int, wz: int) -> float:
        """Read a value; outside the rect the base callback is used."""
        if self.contains(wx, wz):
            return self._array(arr_id)[self._index(wx, wz)]
        if self.external is None:
            raise KeyError(f"cell {(wx, wz)} outside field with no external")
        return self.external(arr_id, wx, wz)

    def set(self, arr_id: int, wx: int, wz: int, value: float) -> None:
        if not self.contains(wx, wz):
            raise KeyError(f"cannot write outside cell {(wx, wz)}")
        self._array(arr_id)[self._index(wx, wz)] = value
