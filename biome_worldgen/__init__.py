"""Deterministic procedural ecological world generation.

The package provides:

* deterministic, seed-based terrain / climate / moisture / vegetation
  generation on a grid of chunks;
* order-independent cross-chunk propagation (synchronous Jacobi iteration)
  with configurable convergence and round cap;
* incremental recomputation of locally modified regions that is guaranteed
  to converge to the same fixed point as a full rebuild;
* concurrent generation with immutable chunks and atomic commits;
* world snapshots, versioned rollback and resumable generation;
* cancellation / timeout with all-or-nothing commits.
"""

from __future__ import annotations

from .config import RegionParams, WorldGenConfig
from .noise import fbm, hash2, value_noise
from .chunk import Cell, Chunk, generate_base_chunk
from .propagation import PropagationResult, propagate
from .snapshot import Snapshot, SnapshotState
from .concurrency import (
    CancelToken,
    OperationCancelled,
    OperationHandle,
    OperationTimeout,
)
from .world import World
from .api import WorldManager, create_app

__version__ = "0.1.0"

__all__ = [
    "RegionParams",
    "WorldGenConfig",
    "fbm",
    "hash2",
    "value_noise",
    "Cell",
    "Chunk",
    "generate_base_chunk",
    "PropagationResult",
    "propagate",
    "Snapshot",
    "SnapshotState",
    "CancelToken",
    "OperationCancelled",
    "OperationHandle",
    "OperationTimeout",
    "World",
    "WorldManager",
    "create_app",
    "__version__",
]
