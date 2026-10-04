"""Domain exceptions."""

from __future__ import annotations


class WorldgenError(Exception):
    """Base class for worldgen errors."""


class CancellationError(WorldgenError):
    """Raised when a generation operation is cancelled or times out."""


class GenerationTimeout(CancellationError):
    """Raised when a generation operation exceeds its deadline."""


class LockAcquireTimeout(WorldgenError):
    """Acquiring chunk locks took longer than the configured timeout."""


class ConcurrentModificationError(WorldgenError):
    """A commit raced with another writer and retries were exhausted."""


class NonConvergedError(WorldgenError):
    """Propagation did not meet the convergence tolerance."""


class SnapshotError(WorldgenError):
    """Invalid snapshot operation (rollback of an open snapshot, ...)."""
