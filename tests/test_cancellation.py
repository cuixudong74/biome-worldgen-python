"""Requirement 6: cancellation and timeout leave no partial writes."""

from __future__ import annotations

import pytest

from worldgen import ChunkCoord, World, WorldConfig
from worldgen.cancellation import CancelToken
from worldgen.config import RegionParams
from worldgen.errors import CancellationError, GenerationTimeout

RECT = (0, 0, 4, 4)
ALL = [ChunkCoord(x, z) for x in range(4) for z in range(4)]


def test_cancelled_before_start_installs_nothing(world):
    token = CancelToken()
    token.cancel()
    with pytest.raises(CancellationError):
        world.generate(rect=RECT, cancel=token)
    assert len(world) == 0
    # Re-trigger succeeds and matches an uninterrupted run.
    world.generate(rect=RECT)
    ref = World(world.config)
    ref.generate(rect=RECT)
    for c in ALL:
        assert world.get_chunk(c).cells == ref.get_chunk(c).cells


def test_cancelled_recompute_installs_nothing(world, config):
    world.generate(rect=RECT)
    before = {c: world.get_chunk(c).cells for c in ALL}
    versions = {c: world.get_chunk(c).version for c in ALL}

    token = CancelToken()
    token.cancel()
    with pytest.raises(CancellationError):
        world.apply_region_params(
            "hot", (1, 1, 3, 3), RegionParams(temperature_bias=0.3),
            cancel=token,
        )

    # No chunk content/version changed despite the registry edit.
    for c in ALL:
        assert world.get_chunk(c).cells == before[c]
        assert world.get_chunk(c).version == versions[c]

    # Retry the same operation and compare to a clean full rebuild.
    world.apply_region_params(
        "hot", (1, 1, 3, 3), RegionParams(temperature_bias=0.3)
    )
    ref = World(config)
    ref.apply_region_params("hot", (1, 1, 3, 3),
                            RegionParams(temperature_bias=0.3),
                            auto_recompute=False)
    ref.generate(rect=RECT)
    for c in ALL:
        assert world.get_chunk(c).cells == ref.get_chunk(c).cells


def test_timeout_aborts_cleanly(config):
    cfg = WorldConfig(
        seed=config.seed,
        max_propagation_rounds=config.max_propagation_rounds,
        generation_timeout_seconds=0.0,
    )
    world = World(cfg)
    with pytest.raises(GenerationTimeout):
        world.generate(rect=RECT)
    assert len(world) == 0
    # A generous deadline afterwards works normally.
    cfg2 = WorldConfig(seed=config.seed,
                       max_propagation_rounds=config.max_propagation_rounds,
                       generation_timeout_seconds=60.0)
    world2 = World(cfg2)
    world2.generate(rect=RECT)
    assert len(world2) == 16


def test_cancel_token_is_idempotent():
    token = CancelToken()
    assert not token.cancelled
    token.cancel()
    token.cancel()
    assert token.cancelled
    with pytest.raises(CancellationError):
        token.check()


def test_interrupted_then_repeated_is_stable(world, config):
    world.generate(rect=RECT)
    # Interrupt once, then complete; interrupt again, then complete again.
    for i in range(2):
        token = CancelToken()
        token.cancel()
        with pytest.raises(CancellationError):
            world.apply_region_params(
                "hot", (0, 0, 2, 2),
                RegionParams(temperature_bias=0.2 + 0.05 * i),
                cancel=token,
            )
    # Finally apply the last value normally.
    world.apply_region_params("hot", (0, 0, 2, 2),
                              RegionParams(temperature_bias=0.25))
    ref = World(config)
    ref.apply_region_params("hot", (0, 0, 2, 2),
                            RegionParams(temperature_bias=0.25),
                            auto_recompute=False)
    ref.generate(rect=RECT)
    for c in ALL:
        assert world.get_chunk(c).cells == ref.get_chunk(c).cells
