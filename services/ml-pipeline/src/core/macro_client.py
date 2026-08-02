"""HTTP client for the macro regime — current (serving) and historical (training).

Training appends a macro one-hot per session date; serving needs the same
context at inference time. The regime moves on a 6-hour refresh cadence, so
a short in-memory TTL cache keeps inference from hammering macro-data on
every features.ready. Failures degrade to None — the all-zeros one-hot the
dataset uses for "unknown".

`get_regime_history` is the P2-4 half: macro-data classifies each past day from
ONLY the vintages published by that day, so the columns it fills are something
a model may legitimately learn from. Before it existed `build_dataset`'s
`regime_by_date` had no caller at all and the five `macro_*` columns were
all-zero in every run.
"""

import time
from collections.abc import Callable
from datetime import date, datetime
from typing import Protocol

import httpx
import structlog

logger = structlog.get_logger()


class MacroClient(Protocol):
    async def get_regime(self) -> str | None: ...

    async def get_regime_history(self, start: date, end: date) -> dict[date, str]: ...

    async def aclose(self) -> None: ...


class NullMacroClient:
    async def get_regime(self) -> str | None:
        return None

    async def get_regime_history(self, start: date, end: date) -> dict[date, str]:
        return {}

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

    async def get_regime_history(self, start: date, end: date) -> dict[date, str]:
        """Regime per calendar day, each classified point-in-time by macro-data.

        NOT cached: a training run asks once for a 20-year window, and caching
        a payload that size to serve one caller would trade memory for nothing.
        An unreachable service yields an empty mapping, which the dataset turns
        into the all-zeros "unknown" one-hot — the same thing it did before this
        endpoint existed, so a macro outage degrades training rather than
        failing it.
        """
        try:
            resp = await self._client.get(
                f"{self._base}/api/v1/macro-data/history",
                params={"start": start.isoformat(), "end": end.isoformat()},
                timeout=120.0,
            )
            resp.raise_for_status()
            payload = resp.json().get("regimes", {})
        except httpx.HTTPError as exc:
            logger.warning("Macro history lookup failed", error=str(exc))
            return {}

        out: dict[date, str] = {}
        for key, value in payload.items():
            if not isinstance(value, str):
                continue
            try:
                out[datetime.strptime(key, "%Y-%m-%d").date()] = value
            except (TypeError, ValueError):
                continue
        logger.info("Macro history fetched", days=len(out), start=str(start), end=str(end))
        return out

    async def aclose(self) -> None:
        await self._client.aclose()
