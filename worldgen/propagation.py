"""Cross-cell relaxation that propagates climate and water.

The operator is a fixed-order synchronous (Jacobi) averaging:

* all cells of one array are updated from the previous round's snapshot
  only, so results are independent of chunk boundaries, chunk generation
  order and concurrent scheduling;
* each new value is a convex combination (weights sum to 1) of the old
  value and its four neighbours, so the update is a contraction in the
  supremum norm - differences shrink every round and propagation always
  terminates. Cycles in the dependency graph cannot cause an infinite
  loop; the number of rounds is bounded by the configurable
  ``max_propagation_rounds``;
* running ``K`` rounds on a rectangle whose outer border is fed with base
  values yields, on the interior farther than ``K`` cells from that border,
  the exact same values as a run on an infinite/full domain. That identity
  is what makes incremental recompute bit-identical to a full rebuild.
"""

from __future__ import annotations

from worldgen.cancellation import CancelToken
from worldgen.fields import (
    ELEVATION,
    Fields,
    MOISTURE,
    TEMPERATURE,
    WATER,
)
from worldgen.terrain import ParamResolver

TEMPERATURE_DIFFUSION: float = 0.20
MOISTURE_DIFFUSION: float = 0.20
WATER_DIFFUSION: float = 0.15
WATER_DOWNHILL: float = 0.05
EPSILON: float = 1e-12


def _neighbour_average(fields: Fields, arr_id: int, wx: int, wz: int) -> float:
    get = fields.get
    return (
        get(arr_id, wx + 1, wz)
        + get(arr_id, wx - 1, wz)
        + get(arr_id, wx, wz + 1)
        + get(arr_id, wx, wz - 1)
    ) * 0.25


def _relax_array(
    fields: Fields,
    arr_id: int,
    diffusion: float,
    downhill: bool,
    new_values: list[float],
) -> float:
    """One synchronous round on one array; returns max |delta|."""
    keep = 1.0 - diffusion
    max_delta = 0.0
    k = 0
    for wz in range(fields.wz0, fields.wz1):
        for wx in range(fields.wx0, fields.wx1):
            old = fields.get(arr_id, wx, wz)
            value = keep * old + diffusion * _neighbour_average(
                fields, arr_id, wx, wz
            )
            if downhill:
                # Water additionally flows from higher elevation to lower,
                # using last round's elevation (immutable) for the slope.
                e = fields.get(ELEVATION, wx, wz)
                e_n = fields.get(ELEVATION, wx + 1, wz)
                e_s = fields.get(ELEVATION, wx - 1, wz)
                e_e = fields.get(ELEVATION, wx, wz + 1)
                e_w = fields.get(ELEVATION, wx, wz - 1)
                slope = ((e - e_n) + (e - e_s) + (e - e_e) + (e - e_w)) / 4.0
                value = max(0.0, value + WATER_DOWNHILL * slope)
            new_values[k] = value
            delta = abs(value - old)
            if delta > max_delta:
                max_delta = delta
            k += 1
    arr = fields._array(arr_id)
    for i, v in enumerate(new_values):
        arr[i] = v
    return max_delta


def propagate(
    fields: Fields,
    *,
    rounds: int,
    tolerance: float,
    resolver: ParamResolver,
    cancel: CancelToken | None = None,
) -> tuple[int, float, bool]:
    """Run Jacobi relaxation over the mutable domain.

    ``resolver`` is accepted for symmetry/future source terms; current
    biases enter through the initial fields and relaxation is homogeneous.

    Returns ``(rounds_executed, max_delta_last_round, converged)``.
    """
    del resolver
    if rounds < 0:
        raise ValueError("rounds must be non-negative")
    size = fields.width * fields.height
    scratch = [0.0] * size
    executed = 0
    last_delta = 0.0
    for r in range(rounds):
        if cancel is not None:
            cancel.check()
        executed = r + 1
        d1 = _relax_array(fields, TEMPERATURE, TEMPERATURE_DIFFUSION, False, scratch)
        d2 = _relax_array(fields, MOISTURE, MOISTURE_DIFFUSION, False, scratch)
        d3 = _relax_array(fields, WATER, WATER_DIFFUSION, True, scratch)
        last_delta = max(d1, d2, d3)
        if tolerance > 0.0 and last_delta <= tolerance:
            return executed, last_delta, True
    converged = last_delta <= tolerance if tolerance > 0.0 else True
    return executed, last_delta, converged
