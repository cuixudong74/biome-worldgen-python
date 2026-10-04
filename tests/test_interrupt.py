"""Cancellation and timeout: interrupted operations leave no partial
writes, and re-triggering afterwards reproduces the correct result."""

from __future__ import annotations

import threading
import time

import pytest

from biome_worldgen import (
    CancelToken,
    OperationCancelled,
    OperationTimeout,
    World,
    WorldGenConfig,
)


def _cfg(**kw) -> WorldGenConfig:
    base = dict(chunk_size=4, max_propagation_rounds=300, convergence_epsilon=1e-12)
    base.update(kw)
    return WorldGenConfig(**base)


def _fingerprint(world: World) -> dict:
    out = {}
    for key in sorted(world.chunk_keys()):
        chunk = world.get_chunk(*key)
        out[key] = tuple(
            (c.height, c.temperature, c.moisture, c.vegetation) for c in chunk.cells
        )
    return out


def test_cancel_before_commit_writes_nothing():
    world = World(seed=1818, config=_cfg())
    token = CancelToken()
    token.cancel()  # pre-cancelled
    before = world.chunk_keys()
    with pytest.raises(OperationCancelled):
        world.generate_region(0, 0, 4, 4, cancel_token=token)
    assert world.chunk_keys() == before  # nothing written


def test_timeout_before_commit_writes_nothing():
    world = World(seed=1919, config=_cfg())
    before = world.chunk_keys()
    with pytest.raises(OperationTimeout):
        world.generate_region(0, 0, 6, 6, timeout=0.001)
    assert world.chunk_keys() == before  # nothing written


def test_cancel_midflight_writes_nothing():
    world = World(seed=2020, config=_cfg())
    token = CancelToken()
    errors: list[BaseException] = []

    def worker():
        try:
            world.generate_region(0, 0, 8, 8, cancel_token=token)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    t = threading.Thread(target=worker)
    t.start()
    time.sleep(0.05)
    token.cancel()
    t.join()
    assert errors and isinstance(errors[0], OperationCancelled)
    # Either nothing was written or the whole region completed; never a
    # partial region.
    keys = set(world.chunk_keys())
    full = {(cx, cz) for cz in range(9) for cx in range(9)}
    assert keys == set() or keys == full, f"partial write: {len(keys)} chunks"


def test_retrigger_after_cancel_is_correct():
    world = World(seed=2121, config=_cfg())
    token = CancelToken()
    token.cancel()
    with pytest.raises(OperationCancelled):
        world.generate_region(0, 0, 4, 4, cancel_token=token)
    # Re-trigger with a fresh token.
    world.generate_region(0, 0, 4, 4)
    ref = World(seed=2121, config=_cfg())
    ref.generate_region(0, 0, 4, 4)
    assert _fingerprint(world) == _fingerprint(ref)


def test_cancel_recompute_writes_nothing():
    world = World(seed=2222, config=_cfg())
    world.generate_region(0, 0, 5, 5)
    before = {k: world.get_chunk(*k) for k in world.chunk_keys()}
    token = CancelToken()
    token.cancel()
    with pytest.raises(OperationCancelled):
        world.recompute({(2, 2): __import__("biome_worldgen").RegionParams(
            temperature_offset=5.0)}, cancel_token=token)
    after = {k: world.get_chunk(*k) for k in world.chunk_keys()}
    assert before == after  # nothing rewritten


def test_timeout_recompute_writes_nothing():
    world = World(seed=2323, config=_cfg())
    world.generate_region(0, 0, 5, 5)
    before = {k: world.get_chunk(*k) for k in world.chunk_keys()}
    with pytest.raises(OperationTimeout):
        world.recompute({(2, 2): __import__("biome_worldgen").RegionParams(
            temperature_offset=5.0)}, timeout=0.001)
    after = {k: world.get_chunk(*k) for k in world.chunk_keys()}
    assert before == after


def test_operation_handle_cancel():
    world = World(seed=2424, config=_cfg())
    handle = world.register_operation("generate")
    assert handle.status == "pending"
    handle.token.cancel()
    with pytest.raises(OperationCancelled):
        world.generate_region(0, 0, 2, 2, cancel_token=handle.token)
    handle.mark_finished("cancelled")
    assert handle.status == "cancelled"
    assert world.chunk_keys() == []
