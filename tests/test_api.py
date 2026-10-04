"""HTTP API smoke tests."""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from biome_worldgen import WorldManager, create_app


def _client() -> TestClient:
    return TestClient(create_app(WorldManager()))


def _wait(client: TestClient, world_id: str, op_id: str, timeout: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = client.get(f"/api/worlds/{world_id}/operations/{op_id}")
        assert r.status_code == 200
        body = r.json()
        if body["status"] in ("succeeded", "cancelled", "failed"):
            return body
        time.sleep(0.05)
    raise AssertionError("operation did not finish in time")


def test_create_world():
    client = _client()
    r = client.post("/api/worlds", json={"seed": 42})
    assert r.status_code == 200
    body = r.json()
    assert body["seed"] == 42
    assert "world_id" in body


def test_generate_and_read_chunk():
    client = _client()
    r = client.post("/api/worlds", json={"seed": 42})
    world_id = r.json()["world_id"]
    r = client.post(f"/api/worlds/{world_id}/generate", json={
        "x0": 0, "z0": 0, "x1": 2, "z1": 2,
    })
    assert r.status_code == 200
    op_id = r.json()["operation_id"]
    body = _wait(client, world_id, op_id)
    assert body["status"] == "succeeded", body
    assert len(body["result"]["added"]) == 9

    r = client.get(f"/api/worlds/{world_id}/chunks/0/0")
    assert r.status_code == 200
    chunk = r.json()
    assert chunk["cx"] == 0
    assert chunk["cz"] == 0
    assert len(chunk["cells"]) == chunk["size"] ** 2


def test_recompute():
    client = _client()
    r = client.post("/api/worlds", json={"seed": 42})
    world_id = r.json()["world_id"]
    r = client.post(f"/api/worlds/{world_id}/generate", json={
        "x0": 0, "z0": 0, "x1": 2, "z1": 2,
    })
    _wait(client, world_id, r.json()["operation_id"])

    r = client.post(f"/api/worlds/{world_id}/recompute", json={
        "changes": {"1,1": {"temperature_offset": 5.0}},
    })
    assert r.status_code == 200
    body = _wait(client, world_id, r.json()["operation_id"])
    assert body["status"] == "succeeded", body
    assert "1,1" in body["result"]["updated"]


def test_snapshot_and_rollback():
    client = _client()
    r = client.post("/api/worlds", json={"seed": 42})
    world_id = r.json()["world_id"]
    r = client.post(f"/api/worlds/{world_id}/generate", json={
        "x0": 0, "z0": 0, "x1": 1, "z1": 1,
    })
    _wait(client, world_id, r.json()["operation_id"])

    r = client.post(f"/api/worlds/{world_id}/snapshots", json={"label": "v1"})
    assert r.status_code == 200
    snap_id = r.json()["snapshot_id"]
    assert r.json()["state"] == "committed"

    r = client.post(f"/api/worlds/{world_id}/generate", json={
        "x0": 2, "z0": 2, "x1": 3, "z1": 3,
    })
    _wait(client, world_id, r.json()["operation_id"])
    assert len(client.get(f"/api/worlds/{world_id}/chunks").json()["chunks"]) == 8

    r = client.post(f"/api/worlds/{world_id}/snapshots/{snap_id}/rollback")
    assert r.status_code == 200
    assert len(client.get(f"/api/worlds/{world_id}/chunks").json()["chunks"]) == 4


def test_cancel_operation():
    client = _client()
    r = client.post("/api/worlds", json={"seed": 42})
    world_id = r.json()["world_id"]
    r = client.post(f"/api/worlds/{world_id}/generate", json={
        "x0": 0, "z0": 0, "x1": 8, "z1": 8,
    })
    op_id = r.json()["operation_id"]
    # Cancel immediately.
    r = client.post(f"/api/worlds/{world_id}/operations/{op_id}/cancel")
    assert r.status_code == 200
    body = _wait(client, world_id, op_id)
    assert body["status"] in ("cancelled", "succeeded")  # may have finished first


def test_404_on_unknown_world():
    client = _client()
    r = client.get("/api/worlds/nope")
    assert r.status_code == 404
    r = client.get("/api/worlds/nope/chunks/0/0")
    assert r.status_code == 404
