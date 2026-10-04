"""Optional HTTP interface over the worldgen engine.

Run with ``uv run uvicorn main:app``. The server holds a single demo
:class:`~worldgen.World`; all write endpoints are concurrency-safe.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from worldgen import ChunkCoord, RegionParams, World, WorldConfig
from worldgen.errors import SnapshotError, WorldgenError


class ConfigIn(BaseModel):
    seed: int = 1337
    max_propagation_rounds: int = Field(default=4, ge=0, le=256)
    convergence_tolerance: float = 0.0
    non_converged_policy: str = "accept"
    generation_timeout_seconds: float | None = None


class RegionIn(BaseModel):
    region_id: str
    rect: tuple[int, int, int, int]
    temperature_bias: float = 0.0
    moisture_bias: float = 0.0
    water_bias: float = 0.0
    elevation_bias: float = 0.0
    border_blend_cells: int = 8
    auto_recompute: bool = True


app = FastAPI(title="biome-worldgen", version="0.1.0")
_world: World = World(WorldConfig())


@app.get("/")
def read_root() -> dict:
    return {
        "service": "biome-worldgen",
        "chunks": len(_world),
        "snapshots": len(_world.list_snapshots()),
    }


@app.post("/world/configure")
def configure(cfg: ConfigIn) -> dict:
    """Reset the demo world with a fresh configuration."""
    global _world
    _world = World(
        WorldConfig(
            seed=cfg.seed,
            max_propagation_rounds=cfg.max_propagation_rounds,
            convergence_tolerance=cfg.convergence_tolerance,
            non_converged_policy=cfg.non_converged_policy,
            generation_timeout_seconds=cfg.generation_timeout_seconds,
        )
    )
    return {"ok": True, "chunks": 0}


@app.post("/generate")
def generate(rect: tuple[int, int, int, int]) -> dict:
    try:
        chunks = _world.generate(rect=rect)
    except (ValueError, WorldgenError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"generated": len(chunks), "total": len(_world)}


@app.get("/chunks/{cx}/{cz}")
def get_chunk(cx: int, cz: int) -> dict:
    chunk = _world.get_chunk(ChunkCoord(cx, cz))
    if chunk is None:
        raise HTTPException(status_code=404, detail="chunk not generated")
    return {
        "coord": [chunk.coord.cx, chunk.coord.cz],
        "version": chunk.version,
        "params_digest": chunk.params_digest,
        "cells": [
            {
                "elevation": c.elevation,
                "temperature": c.temperature,
                "moisture": c.moisture,
                "water": c.water,
                "biome": c.biome.value,
            }
            for c in chunk.cells
        ],
    }


@app.post("/regions")
def apply_region(region: RegionIn) -> dict:
    params = RegionParams(
        temperature_bias=region.temperature_bias,
        moisture_bias=region.moisture_bias,
        water_bias=region.water_bias,
        elevation_bias=region.elevation_bias,
        border_blend_cells=region.border_blend_cells,
    )
    try:
        changed = _world.apply_region_params(
            region.region_id,
            region.rect,
            params,
            auto_recompute=region.auto_recompute,
        )
    except (ValueError, WorldgenError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"recomputed": len(changed), "total": len(_world)}


@app.delete("/regions/{region_id}")
def remove_region(region_id: str, auto_recompute: bool = True) -> dict:
    changed = _world.remove_region(
        region_id, auto_recompute=auto_recompute
    )
    return {"recomputed": len(changed), "total": len(_world)}


@app.post("/recompute")
def recompute() -> dict:
    changed = _world.recompute()
    return {"recomputed": len(changed)}


@app.post("/snapshots")
def create_snapshot(note: str = "") -> dict:
    snap = _world.snapshot(note)
    return {"snapshot_id": snap.snapshot_id, "state": snap.state.value}


@app.get("/snapshots")
def list_snapshots() -> dict:
    return {
        "snapshots": [
            {
                "id": s.snapshot_id,
                "state": s.state.value,
                "note": s.note,
                "chunks": len(s.chunks),
            }
            for s in _world.list_snapshots()
        ]
    }


@app.post("/rollback/{snapshot_id}")
def rollback(snapshot_id: str) -> dict:
    try:
        _world.rollback(snapshot_id)
    except SnapshotError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="unknown snapshot") from exc
    return {"ok": True, "chunks": len(_world)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
