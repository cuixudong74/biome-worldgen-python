"""HTTP API for the world generator (FastAPI).

Long running generation / recompute requests run in background threads;
each request is tracked as an :class:`OperationHandle` and can be
cancelled.  Reads always return complete, immutable chunks.
"""

from __future__ import annotations

import threading
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .concurrency import OperationCancelled, OperationTimeout
from .config import RegionParams, WorldGenConfig
from .world import World


class WorldManager:
    """In-memory registry of worlds."""

    def __init__(self) -> None:
        self._worlds: dict[str, World] = {}
        self._lock = threading.RLock()

    def create_world(self, seed: int, config: WorldGenConfig | None = None) -> tuple[str, World]:
        import uuid

        world = World(seed=seed, config=config)
        world_id = uuid.uuid4().hex
        with self._lock:
            self._worlds[world_id] = world
        return world_id, world

    def get_world(self, world_id: str) -> World:
        with self._lock:
            world = self._worlds.get(world_id)
        if world is None:
            raise KeyError(world_id)
        return world

    def list_worlds(self) -> list[str]:
        with self._lock:
            return list(self._worlds.keys())


# ----------------------------------------------------------------------
# request models
# ----------------------------------------------------------------------
class CreateWorldRequest(BaseModel):
    seed: int = 42
    chunk_size: int | None = None
    max_propagation_rounds: int | None = None
    convergence_epsilon: float | None = None
    propagation_enabled: bool | None = None


class GenerateRequest(BaseModel):
    x0: int
    z0: int
    x1: int
    z1: int
    temperature_offset: float = 0.0
    moisture_offset: float = 0.0
    fertility: float = 1.0
    timeout: float | None = None


class RecomputeRequest(BaseModel):
    changes: dict[str, dict[str, float]]
    timeout: float | None = None


class SnapshotRequest(BaseModel):
    label: str | None = None


def _run_operation(world: World, handle, fn) -> None:
    try:
        handle.status = "running"
        result = fn()
        handle.mark_finished("succeeded", result=result)
    except OperationCancelled as exc:
        handle.mark_finished("cancelled", error=str(exc))
    except OperationTimeout as exc:
        handle.mark_finished("failed", error=str(exc))
    except Exception as exc:  # noqa: BLE001
        handle.mark_finished("failed", error=f"{type(exc).__name__}: {exc}")


def create_app(manager: WorldManager | None = None) -> FastAPI:
    manager = manager or WorldManager()
    app = FastAPI(title="Biome World Generator", version="0.1.0")

    @app.get("/")
    def root() -> dict[str, Any]:
        return {"service": "biome-worldgen", "worlds": manager.list_worlds()}

    @app.post("/api/worlds")
    def create_world(req: CreateWorldRequest) -> dict[str, Any]:
        cfg = WorldGenConfig(
            **{k: v for k, v in (
                ("chunk_size", req.chunk_size),
                ("max_propagation_rounds", req.max_propagation_rounds),
                ("convergence_epsilon", req.convergence_epsilon),
                ("propagation_enabled", req.propagation_enabled),
            ) if v is not None}
        ) if any(v is not None for v in (
            req.chunk_size, req.max_propagation_rounds,
            req.convergence_epsilon, req.propagation_enabled,
        )) else WorldGenConfig()
        world_id, _ = manager.create_world(seed=req.seed, config=cfg)
        return {"world_id": world_id, "seed": req.seed}

    @app.get("/api/worlds")
    def list_worlds() -> dict[str, Any]:
        return {"worlds": manager.list_worlds()}

    @app.get("/api/worlds/{world_id}")
    def get_world(world_id: str) -> dict[str, Any]:
        try:
            world = manager.get_world(world_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="world not found")
        return {
            "world_id": world_id,
            "seed": world.seed,
            "version": world.version,
            "chunks": len(world.chunk_keys()),
            "current_snapshot_id": world.current_snapshot_id,
        }

    @app.post("/api/worlds/{world_id}/generate")
    def generate(world_id: str, req: GenerateRequest) -> dict[str, Any]:
        try:
            world = manager.get_world(world_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="world not found")
        handle = world.register_operation("generate")
        params = RegionParams(
            temperature_offset=req.temperature_offset,
            moisture_offset=req.moisture_offset,
            fertility=req.fertility,
        )
        thread = threading.Thread(
            target=_run_operation,
            args=(world, handle, lambda: world.generate_region(
                req.x0, req.z0, req.x1, req.z1, params,
                cancel_token=handle.token, timeout=req.timeout,
            )),
            daemon=True,
        )
        thread.start()
        return {"operation_id": handle.id, "status": handle.status}

    @app.post("/api/worlds/{world_id}/recompute")
    def recompute(world_id: str, req: RecomputeRequest) -> dict[str, Any]:
        try:
            world = manager.get_world(world_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="world not found")
        handle = world.register_operation("recompute")
        changes = {}
        for key, values in req.changes.items():
            cx_str, cz_str = key.split(",")
            changes[(int(cx_str), int(cz_str))] = RegionParams(
                temperature_offset=float(values.get("temperature_offset", 0.0)),
                moisture_offset=float(values.get("moisture_offset", 0.0)),
                fertility=float(values.get("fertility", 1.0)),
            )
        thread = threading.Thread(
            target=_run_operation,
            args=(world, handle, lambda: world.recompute(
                changes, cancel_token=handle.token, timeout=req.timeout,
            )),
            daemon=True,
        )
        thread.start()
        return {"operation_id": handle.id, "status": handle.status}

    @app.get("/api/worlds/{world_id}/operations/{operation_id}")
    def get_operation(world_id: str, operation_id: str) -> dict[str, Any]:
        try:
            world = manager.get_world(world_id)
            handle = world.get_operation(operation_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="not found")
        return {
            "operation_id": handle.id,
            "kind": handle.kind,
            "status": handle.status,
            "result": _result_to_json(handle.result),
            "error": handle.error,
        }

    @app.post("/api/worlds/{world_id}/operations/{operation_id}/cancel")
    def cancel_operation(world_id: str, operation_id: str) -> dict[str, Any]:
        try:
            world = manager.get_world(world_id)
            handle = world.get_operation(operation_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="not found")
        handle.token.cancel()
        return {"operation_id": handle.id, "status": "cancelling"}

    @app.get("/api/worlds/{world_id}/chunks")
    def list_chunks(world_id: str) -> dict[str, Any]:
        try:
            world = manager.get_world(world_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="world not found")
        return {"chunks": [f"{cx},{cz}" for cx, cz in world.chunk_keys()]}

    @app.get("/api/worlds/{world_id}/chunks/{cx}/{cz}")
    def get_chunk(world_id: str, cx: int, cz: int) -> dict[str, Any]:
        try:
            world = manager.get_world(world_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="world not found")
        chunk = world.get_chunk(cx, cz)
        if chunk is None:
            raise HTTPException(status_code=404, detail="chunk not generated")
        return _chunk_to_json(chunk)

    @app.post("/api/worlds/{world_id}/snapshots")
    def create_snapshot(world_id: str, req: SnapshotRequest) -> dict[str, Any]:
        try:
            world = manager.get_world(world_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="world not found")
        snap = world.create_snapshot(label=req.label)
        return {"snapshot_id": snap.id, "state": snap.state.value, "version": snap.version}

    @app.get("/api/worlds/{world_id}/snapshots")
    def list_snapshots(world_id: str) -> dict[str, Any]:
        try:
            world = manager.get_world(world_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="world not found")
        return {
            "snapshots": [
                {
                    "snapshot_id": s.id,
                    "label": s.label,
                    "state": s.state.value,
                    "version": s.version,
                    "parent_id": s.parent_id,
                }
                for s in world.list_snapshots()
            ]
        }

    @app.post("/api/worlds/{world_id}/snapshots/{snapshot_id}/rollback")
    def rollback(world_id: str, snapshot_id: str) -> dict[str, Any]:
        try:
            world = manager.get_world(world_id)
            snap = world.rollback(snapshot_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="not found")
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        return {"snapshot_id": snap.id, "state": snap.state.value, "version": world.version}

    return app


def _chunk_to_json(chunk) -> dict[str, Any]:
    return {
        "cx": chunk.cx,
        "cz": chunk.cz,
        "size": chunk.size,
        "version": chunk.version,
        "cells": [
            {
                "height": c.height,
                "temperature": c.temperature,
                "moisture": c.moisture,
                "vegetation": c.vegetation,
            }
            for c in chunk.cells
        ],
    }


def _result_to_json(result) -> dict[str, Any] | None:
    if result is None:
        return None
    if hasattr(result, "added_keys"):
        return {
            "added": [f"{cx},{cz}" for cx, cz in result.added_keys],
            "updated": [f"{cx},{cz}" for cx, cz in result.updated_keys],
            "rounds": result.rounds,
            "converged": result.converged,
            "max_delta": result.max_delta,
        }
    return {
        "updated": [f"{cx},{cz}" for cx, cz in result.updated_keys],
        "rounds": result.rounds,
        "converged": result.converged,
        "max_delta": result.max_delta,
    }
