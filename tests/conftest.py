"""Shared pytest fixtures and world assertion helpers."""

from __future__ import annotations

import pytest

from worldgen import World, WorldConfig
from worldgen.config import RegionParams


@pytest.fixture
def config() -> WorldConfig:
    # Deterministic seed; 8 propagation rounds (1-chunk influence radius).
    return WorldConfig(seed=20240917, max_propagation_rounds=4)


@pytest.fixture
def world(config: WorldConfig) -> World:
    return World(config)


@pytest.fixture
def make_world(config):
    def _make():
        return World(config)

    return _make


HOT = RegionParams(temperature_bias=0.3)
WET = RegionParams(moisture_bias=0.3, water_bias=0.1)
COLD = RegionParams(temperature_bias=-0.3)


def all_cells_equal(a: World, b: World, coords) -> None:
    """Assert two worlds agree cell-for-cell over ``coords``."""
    for coord in coords:
        ca = a.get_chunk(coord)
        cb = b.get_chunk(coord)
        assert ca is not None and cb is not None, coord
        assert ca.cells == cb.cells, coord
