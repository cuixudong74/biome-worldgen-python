"""Cooperative cancellation and deadline support.

Generation checks the token at well-defined phase boundaries (before each
propagation round and before each chunk materialisation). On cancel/timeout
the pass simply stops producing output and never installs anything, so a
re-triggered request starts from the last committed state.
"""

from __future__ import annotations

import threading
import time

from worldgen.errors import CancellationError, GenerationTimeout


class CancelToken:
    """Thread-safe cooperative cancellation flag with optional deadline."""

    __slots__ = ("_event", "_deadline")

    def __init__(
        self, deadline: float | None = None, *, _event=None
    ) -> None:
        self._event = _event if _event is not None else threading.Event()
        self._deadline = deadline

    @classmethod
    def with_timeout(cls, seconds: float | None) -> "CancelToken":
        deadline = None if seconds is None else time.monotonic() + seconds
        return cls(deadline)

    @property
    def expired(self) -> bool:
        return self._deadline is not None and time.monotonic() >= self._deadline

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set() or self.expired

    def check(self) -> None:
        """Raise :class:`CancellationError` if cancelled/expired."""
        if self._event.is_set():
            raise CancellationError("generation cancelled")
        if self.expired:
            raise GenerationTimeout("generation timed out")
