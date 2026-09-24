"""In-process sliding-window rate limit for the LLM-backed chat endpoint (Phase 12).

Keyed by (verified subject, tenant) so one user cannot burn the hosted model budget and
cannot affect other users. Single-process by design **(V)**: the demo runs one API
instance; a multi-instance deployment needs a shared store (docs/pending-items.md) - no
Redis is added without a measured requirement.
"""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable

from app.core.errors import AppError


class RateLimitedError(AppError):
    status_code = 429
    code = "rate_limited"
    message = "Too many requests. Please wait a moment and try again."

    def __init__(self, retry_after: float) -> None:
        super().__init__()
        self.headers = {"Retry-After": str(max(1, math.ceil(retry_after)))}


class RateLimiter:
    def __init__(
        self,
        limit: int,
        *,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        max_keys: int = 10_000,
    ) -> None:
        self.limit = limit
        self.window = window_seconds
        self._clock = clock
        self._max_keys = max_keys
        self._hits: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = threading.Lock()

    def hit(self, key: str) -> float | None:
        """Record one request; return seconds to wait if the limit is exceeded, else None."""
        if self.limit <= 0:
            return None
        now = self._clock()
        with self._lock:
            self._evict(now)
            hits = self._hits.setdefault(key, deque())
            self._hits.move_to_end(key)
            while hits and hits[0] <= now - self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return hits[0] + self.window - now
            hits.append(now)
            return None

    def check(self, key: str) -> None:
        retry = self.hit(key)
        if retry is not None:
            raise RateLimitedError(retry)

    def tracked_keys(self) -> int:
        return len(self._hits)

    def _evict(self, now: float) -> None:
        # Drop idle keys (oldest first), then enforce the hard bound.
        while self._hits:
            key, hits = next(iter(self._hits.items()))
            if hits and hits[-1] > now - self.window:
                break
            del self._hits[key]
        while len(self._hits) >= self._max_keys:
            self._hits.popitem(last=False)


def limiter_for(app) -> RateLimiter:
    limiter = getattr(app.state, "rate_limiter", None)
    if limiter is None:
        limiter = RateLimiter(app.state.settings.agent_rate_limit_per_minute)
        app.state.rate_limiter = limiter
    return limiter
