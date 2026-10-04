"""Concurrency primitives: cancellation tokens and operation handles."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any


class OperationCancelled(Exception):
    """Raised when an operation is cancelled before it could commit."""


class OperationTimeout(Exception):
    """Raised when an operation passes its deadline before committing."""


class CancelToken:
    """Thread-safe cancellation token.

    Long running operations check :meth:`raise_if_cancelled` between rounds
    and between chunks; the commit phase is only entered while the token is
    still valid, so a cancelled operation never leaves partial writes.
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise OperationCancelled("operation was cancelled")


def deadline_from_timeout(timeout: float | None) -> float | None:
    if timeout is None or timeout <= 0:
        return None
    return time.monotonic() + timeout


def check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() > deadline:
        raise OperationTimeout("operation timed out")


@dataclass
class OperationHandle:
    """Bookkeeping for an in-flight or finished generation operation."""

    id: str
    kind: str
    token: CancelToken
    status: str = "pending"  # pending | running | succeeded | cancelled | failed
    started_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None
    result: Any = None
    error: str | None = None

    def mark_finished(self, status: str, result: Any = None, error: str | None = None) -> None:
        self.status = status
        self.result = result
        self.error = error
        self.finished_at = time.monotonic()
