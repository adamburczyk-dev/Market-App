"""FRED (Federal Reserve Economic Data) HTTP client.

Fetches the latest observation for a small set of macro series over the public
FRED API. Requires ``FRED_API_KEY``; without one the client is disabled and every
fetch returns ``None`` (the service then relies on manually-posted indicators).
Values reported as "." by FRED (missing) are normalized to ``None``.
"""

import asyncio
from typing import Protocol

import httpx
import structlog

logger = structlog.get_logger()

# Indicator name → FRED series id. All are published directly by FRED.
DEFAULT_SERIES = {
    "yield_curve_10y_2y": "T10Y2Y",  # 10Y minus 2Y Treasury spread
    "credit_spread_baa_10y": "BAA10Y",  # Moody's BAA minus 10Y Treasury
    "unemployment_rate": "UNRATE",
    "fed_funds_rate": "FEDFUNDS",
}


class MacroFetcher(Protocol):
    @property
    def enabled(self) -> bool: ...

    async def fetch_indicators(self) -> dict[str, float | None]: ...

    async def aclose(self) -> None: ...


class FredClient:
    def __init__(
        self,
        api_key: str | None,
        series: dict[str, str] | None = None,
        base_url: str = "https://api.stlouisfed.org/fred",
        timeout_s: float = 10.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._series = series or dict(DEFAULT_SERIES)
        self._base = base_url.rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=timeout_s)

    @property
    def enabled(self) -> bool:
        return bool(self._api_key)

    async def latest(self, series_id: str) -> float | None:
        if not self._api_key:
            return None
        params: dict[str, str | int] = {
            "series_id": series_id,
            "api_key": self._api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 1,
        }
        try:
            resp = await self._client.get(f"{self._base}/series/observations", params=params)
            resp.raise_for_status()
            obs = resp.json().get("observations", [])
        except httpx.HTTPError as exc:
            logger.warning("FRED fetch failed", series_id=series_id, error=str(exc))
            return None
        if not obs:
            return None
        value = obs[0].get("value")
        if value in (None, ".", ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    async def fetch_indicators(self) -> dict[str, float | None]:
        """Fetch every configured series concurrently → indicator name → value/None."""
        names = list(self._series)
        values = await asyncio.gather(*(self.latest(self._series[n]) for n in names))
        return dict(zip(names, values, strict=True))

    async def aclose(self) -> None:
        await self._client.aclose()
