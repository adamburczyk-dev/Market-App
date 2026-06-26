"""HTTP client for querying latest prices from market-data (to mark positions)."""

from typing import Protocol

import httpx
import structlog
from trading_common.schemas import Interval, OHLCVBar

logger = structlog.get_logger()


class MarketDataClient(Protocol):
    async def latest_close(self, symbol: str, interval: Interval) -> float | None: ...


class HttpMarketDataClient:
    def __init__(self, base_url: str, timeout_s: float = 5.0) -> None:
        self._base = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout_s)

    async def latest_close(self, symbol: str, interval: Interval) -> float | None:
        url = f"{self._base}/api/v1/market-data/ohlcv/{symbol}"
        resp = await self._client.get(url, params={"interval": interval.value, "limit": 1})
        resp.raise_for_status()
        bars = resp.json()
        if not bars:
            return None
        return float(OHLCVBar.model_validate(bars[-1]).close)

    async def aclose(self) -> None:
        await self._client.aclose()
