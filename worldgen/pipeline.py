"""Pure generation / recomputation pipeline (no global state, no locks).

Given a set of chunks to *write* and the set of chunks that currently
exist, :func:`run_pass` produces the exact chunks that a full rebuild of
those chunks would produce, regardless of which neighbouring chunks were
generated before.

How the identity is achieved
----------------------------
1. Base fields are pure functions of ``(seed, world cell, region params)``;
   nothing is read from already committed neighbours.
2. Relaxation is a fixed-order synchronous Jacobi update. After ``K`` rounds
   the value at a cell only depends on cells within Chebyshev distance ``K``.
3. The pass therefore simulates a cell rectangle enlarged by ``K`` cells
   around the write set (the *halo*), feeding its outer border with base
   values on demand. Interior write chunks are farther than ``K`` cells from
   that border, so their result equals the infinite/full-domain result.

Only the requested write chunks are materialised; halo chunks are never
returned, hence unaffected chunks are never rewritten.
"""

from __future__ import annotations

from worldgen.chunk import Chunk
from worldgen.config import WorldConfig
from worldgen.coords import CHUNK_SIZE, ChunkCoord, chunks_in_rect
from worldgen.fields import Fields
from worldgen.propagation import propagate
from worldgen.rng import RNG
from worldgen.terrain import (
    ParamResolver,
    base_field,
    effective_params,
    initialize_fields,
)
from worldgen.vegetation import build_chunk_cells


def _bounds(coords) -> tuple[int, int, int, int]:
    cxs = [c.cx for c in coords]
    czs = [c.cz for c in coords]
    return min(cxs), min(czs), max(cxs) + 1, max(czs) + 1


def affected_chunk_radius(max_rounds: int) -> int:
    """How many chunks a parameter change can influence.

    ``K`` rounds cross at most ``K`` cell borders; that reaches at most
    ``ceil(K / CHUNK_SIZE)`` (normally one) chunk outward, and we always
    keep a full chunk halo for simulation regardless.
    """
    return max(1, (max_rounds + CHUNK_SIZE - 1) // CHUNK_SIZE)


def affected_chunks(
    changed: set[ChunkCoord], radius: int
) -> set[ChunkCoord]:
    """Expand a changed chunk set by ``radius`` chunks on every side."""
    if not changed:
        return set()
    cx0, cz0, cx1, cz1 = _bounds(changed)
    return set(
        chunks_in_rect(cx0 - radius, cz0 - radius, cx1 + radius, cz1 + radius)
    )


def run_pass(
    *,
    write_coords: set[ChunkCoord],
    config: WorldConfig,
    resolver: ParamResolver,
    params_digest: str,
    cancel=None,
) -> list[Chunk]:
    """Generate the write set deterministically.

    ``write_coords`` must already be intersected with existing chunks for a
    recompute (the world layer filters it); for a first generation it is
    exactly the requested region.
    """
    if not write_coords:
        return []

    rounds = config.max_propagation_rounds
    cx0, cz0, cx1, cz1 = _bounds(write_coords)

    # Cell-space write rectangle and the K-cell-deep simulation halo.
    wx0, wz0 = cx0 * CHUNK_SIZE, cz0 * CHUNK_SIZE
    wx1, wz1 = cx1 * CHUNK_SIZE, cz1 * CHUNK_SIZE
    sim = Fields(
        wx0 - rounds,
        wz0 - rounds,
        wx1 + rounds,
        wz1 + rounds,
        [],
        [],
        [],
        [],
        external=lambda arr, x, z: base_field(arr, config.seed, x, z)
        + _bias_for(arr, resolver, x, z),
    )
    initialize_fields(sim, config.seed, resolver, RNG(config.seed))

    executed, last_delta, converged = propagate(
        sim,
        rounds=rounds,
        tolerance=config.convergence_tolerance,
        resolver=resolver,
        cancel=cancel,
    )
    if (
        not converged
        and config.non_converged_policy == "raise"
        and config.convergence_tolerance > 0.0
    ):
        from worldgen.errors import NonConvergedError

        raise NonConvergedError(
            f"propagation did not converge after {executed} rounds "
            f"(max delta {last_delta!r} > {config.convergence_tolerance!r})"
        )

    ordered = sorted(write_coords)
    chunks: list[Chunk] = []
    rng = RNG(config.seed)
    for coord in ordered:
        if cancel is not None:
            cancel.check()
        cells = build_chunk_cells(
            sim, coord.cx, coord.cz, resolver, config.seed, rng
        )
        # Version 0 placeholder; the store assigns the committed version.
        chunks.append(
            Chunk(
                coord=coord,
                version=0,
                params_digest=params_digest,
                cells=cells,
            )
        )
    return chunks


def _bias_for(arr_id: int, resolver: ParamResolver, wx: int, wz: int) -> float:
    from worldgen.fields import ELEVATION, MOISTURE, TEMPERATURE, WATER

    p = effective_params(resolver, wx, wz)
    if arr_id == ELEVATION:
        return p.elevation_bias
    if arr_id == TEMPERATURE:
        return p.temperature_bias
    if arr_id == MOISTURE:
        return p.moisture_bias
    if arr_id == WATER:
        return p.water_bias
    raise ValueError(arr_id)
