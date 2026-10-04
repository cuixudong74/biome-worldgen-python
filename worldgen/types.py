"""Shared enumerations."""

from __future__ import annotations

from enum import Enum


class Biome(str, Enum):
    """Biome classification for a cell."""

    DEEP_OCEAN = "deep_ocean"
    OCEAN = "ocean"
    BEACH = "beach"
    DESERT = "desert"
    SAVANNA = "savanna"
    GRASSLAND = "grassland"
    FOREST = "forest"
    RAINFOREST = "rainforest"
    TAIGA = "taiga"
    TUNDRA = "tundra"
    SNOW = "snow"
    MOUNTAIN = "mountain"


class SnapshotState(str, Enum):
    """Lifecycle state of a world snapshot."""

    IN_PROGRESS = "in_progress"
    COMMITTED = "committed"
    ABORTED = "aborted"
