"""HTTP client for the current macro regime (training/serving parity).

Training appends a macro one-hot per session date; serving needs the same
context at inference time. The regime moves on a 6-hour refresh cadence, so
a short in-memory TTL cache keeps inference from hammering macro-data on
every features.ready. Failures degrade to None — the all-zeros one-hot the
dataset uses for "unknown".
"""

import time
from collections.abc import Callable
from typing import Protocol

import httpx
import structlog

logger = structlog.get_logger()


class MacroClient(Protocol):
    async def get_regime(self) -> str | None: ...

    async def aclose(self) -> None: ...


class NullMacroClient:
    async def get_regime(self) -> str | None:
        return None

    async def aclose(self) -> None:
        return None


class HttpMacroClient:
    def __init__(
        self,
        base_url: str,
        timeout_s: float = 5.0,
        cache_ttl_s: float = 600.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout_s)
        self._ttl = cache_ttl_s
        self._clock = clock
        self._cached_at: float | None = None
        self._cached: str | None = None

    async def get_regime(self) -> str | None:
        now = self._clock()
        if self._cached_at is not None and now - self._cached_at < self._ttl:
            return self._cached
        regime: str | None = None
        try:
            resp = await self._client.get(f"{self._base}/api/v1/macro-data/regime")
            resp.raise_for_status()
            value = resp.json().get("regime")
            regime = value if isinstance(value, str) else None
        except httpx.HTTPError as exc:
            logger.warning("Macro regime lookup failed", error=str(exc))
        self._cached = regime  # negative results cached too — don't hammer a down service
        self._cached_at = now
        return regime

    async def aclose(self) -> None:
        await self._client.aclose()
