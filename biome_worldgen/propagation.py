"""Cross-chunk propagation.

Climate, moisture and vegetation are relaxed across chunk seams using
*synchronous Jacobi iteration*: every round reads the previous round's
values (including cells owned by neighbouring chunks) and writes a full
new generation.  Because the update is contractive and order independent,
the iteration always converges to a unique fixed point regardless of the
order in which chunks were generated or how many threads participated.

Convergence is reached when the maximum per-cell field change of a round
drops below ``config.convergence_epsilon``, or when
``config.max_propagation_rounds`` is exhausted (safety cap; the system is
contractive so it never deadlocks).
"""

from __future__ import annotations

from dataclasses import dataclass

from .chunk import Cell, Chunk
from .concurrency import CancelToken, check_deadline
from .config import WorldGenConfig


@dataclass
class PropagationResult:
    chunks: dict[tuple[int, int], Chunk]
    rounds: int
    converged: bool
    max_delta: float


def _vegetation_potential(temperature: float, moisture: float, fertility: float) -> float:
    if temperature < 5.0:
        temp_factor = 0.0
    elif temperature < 20.0:
        temp_factor = (temperature - 5.0) / 15.0
    else:
        temp_factor = 1.0
    if moisture < 0.2:
        moist_factor = 0.0
    elif moisture < 0.5:
        moist_factor = (moisture - 0.2) / 0.3
    else:
        moist_factor = 1.0
    return max(0.0, min(1.0, temp_factor * moist_factor * fertility))


def _neighbour_cell(
    snapshot: dict[tuple[int, int], Chunk],
    size: int,
    cx: int,
    cz: int,
    lx: int,
    lz: int,
    dx: int,
    dz: int,
) -> Cell | None:
    """Return the cell offset by ``(dx, dz)`` from ``(lx, lz)``.

    Crosses chunk seams transparently; returns ``None`` when the
    neighbouring chunk has not been loaded yet (zero-flux boundary).
    """
    wx = cx * size + lx + dx
    wz = cz * size + lz + dz
    ncx = wx // size
    ncz = wz // size
    nlx = wx - ncx * size
    nlz = wz - ncz * size
    chunk = snapshot.get((ncx, ncz))
    if chunk is None:
        return None
    return chunk.cell(nlx, nlz)


def _step_cell(
    cell: Cell,
    upwind: Cell | None,
    neighbours: list[Cell],
    config: WorldGenConfig,
    base_moisture: float,
    fertility: float,
) -> tuple[Cell, float]:
    """Compute the next-round cell and its maximum field delta."""
    h = cell.height
    ocean = h < config.sea_level

    # ---- moisture ----------------------------------------------------
    moist = cell.moisture
    if upwind is not None:
        moist += config.moisture_advection * (upwind.moisture - moist)
    if neighbours:
        avg_m = sum(n.moisture for n in neighbours) / len(neighbours)
        moist += config.moisture_diffusion * (avg_m - moist)
    if ocean:
        moist += config.ocean_evaporation * (1.0 - moist)
    else:
        moist += config.groundwater_recharge * (base_moisture - moist)
        elevated = max(0.0, h - config.sea_level)
        moist -= config.rain_shadow * elevated * moist
    moist = max(0.0, min(1.0, moist))

    # ---- temperature -------------------------------------------------
    temp = cell.temperature
    if neighbours:
        avg_t = sum(n.temperature for n in neighbours) / len(neighbours)
        temp += config.temperature_diffusion * (avg_t - temp)
    temp += config.temperature_equilibration * (cell.base_temperature - temp)

    # ---- vegetation --------------------------------------------------
    veg = cell.vegetation
    if neighbours:
        avg_v = sum(n.vegetation for n in neighbours) / len(neighbours)
        veg += config.vegetation_spread * (avg_v - veg)
    potential = _vegetation_potential(temp, moist, fertility)
    veg += config.vegetation_growth * (potential - veg)
    veg = max(0.0, min(1.0, veg))

    new_cell = Cell(
        height=h,
        temperature=temp,
        moisture=moist,
        vegetation=veg,
        base_moisture=base_moisture,
        base_temperature=cell.base_temperature,
    )
    delta = max(
        abs(temp - cell.temperature),
        abs(moist - cell.moisture),
        abs(veg - cell.vegetation),
    )
    return new_cell, delta


def propagate(
    config: WorldGenConfig,
    chunks: dict[tuple[int, int], Chunk],
    cancel_token: CancelToken | None = None,
    deadline: float | None = None,
) -> PropagationResult:
    """Relax all loaded chunks to their cross-chunk fixed point.

    Returns a mapping of ``key -> Chunk`` where unchanged chunks are the
    *same object* (identity preserved) and changed chunks are new
    immutable instances with an incremented version.
    """
    if not config.propagation_enabled:
        return PropagationResult(chunks=dict(chunks), rounds=0, converged=True, max_delta=0.0)

    size = config.chunk_size
    wind_dx = config.wind_dx
    wind_dz = config.wind_dz
    current: dict[tuple[int, int], Chunk] = dict(chunks)
    converged = False
    max_delta = 0.0
    rounds = 0

    for round_no in range(1, config.max_propagation_rounds + 1):
        if cancel_token is not None:
            cancel_token.raise_if_cancelled()
        check_deadline(deadline)

        # Snapshot the previous round: reads never observe a partial write.
        snapshot = current
        next_map: dict[tuple[int, int], Chunk] = {}
        round_delta = 0.0

        for (cx, cz), chunk in snapshot.items():
            new_cells: list[Cell] = []
            changed = False
            for lz in range(size):
                for lx in range(size):
                    cell = chunk.cell(lx, lz)
                    upwind = _neighbour_cell(snapshot, size, cx, cz, lx, lz,
                                             -wind_dx, -wind_dz)
                    neighbour_cells = [
                        n for n in (
                            _neighbour_cell(snapshot, size, cx, cz, lx, lz, 1, 0),
                            _neighbour_cell(snapshot, size, cx, cz, lx, lz, -1, 0),
                            _neighbour_cell(snapshot, size, cx, cz, lx, lz, 0, 1),
                            _neighbour_cell(snapshot, size, cx, cz, lx, lz, 0, -1),
                        ) if n is not None
                    ]
                    new_cell, delta = _step_cell(
                        cell, upwind, neighbour_cells, config,
                        base_moisture=cell.base_moisture,
                        fertility=chunk.params.fertility,
                    )
                    round_delta = max(round_delta, delta)
                    if delta > 0.0:
                        changed = True
                    new_cells.append(new_cell)

            if changed:
                next_map[(cx, cz)] = chunk.with_cells(tuple(new_cells))
            else:
                next_map[(cx, cz)] = chunk

        current = next_map
        max_delta = round_delta
        rounds = round_no
        if round_delta < config.convergence_epsilon:
            converged = True
            break

    # Round final values for deterministic, order-independent commits.
    # Identity is checked against the *original* input chunks so that a
    # converged re-propagation keeps the same objects (no rewrite).
    result: dict[tuple[int, int], Chunk] = {}
    for key, chunk in current.items():
        rounded = tuple(
            Cell(
                height=c.height,
                temperature=round(c.temperature, config.field_decimals),
                moisture=round(c.moisture, config.field_decimals),
                vegetation=round(c.vegetation, config.field_decimals),
                base_moisture=c.base_moisture,
                base_temperature=c.base_temperature,
            )
            for c in chunk.cells
        )
        original = chunks.get(key)
        if original is not None and rounded == original.cells:
            result[key] = original
        else:
            result[key] = chunk.with_cells(rounded)

    return PropagationResult(
        chunks=result, rounds=rounds, converged=converged, max_delta=max_delta
    )
