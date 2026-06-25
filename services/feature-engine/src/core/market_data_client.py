"""HTTP client for querying OHLCV from the market-data service."""

from typing import Protocol

import httpx
import structlog
from trading_common.schemas import Interval, OHLCVBar

logger = structlog.get_logger()


class MarketDataClient(Protocol):
    async def get_ohlcv(
        self, symbol: str, interval: Interval, limit: int = 250
    ) -> list[OHLCVBar]: ...


class HttpMarketDataClient:
    """Queries market-data over HTTP (per architecture: events in, queries via HTTP)."""

    def __init__(self, base_url: str, timeout_s: float = 10.0) -> None:
        self._base = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout_s)

    async def get_ohlcv(
        self, symbol: str, interval: Interval, limit: int = 250
    ) -> list[OHLCVBar]:
        url = f"{self._base}/api/v1/market-data/ohlcv/{symbol}"
        resp = await self._client.get(url, params={"interval": interval.value, "limit": limit})
        resp.raise_for_status()
        return [OHLCVBar.model_validate(item) for item in resp.json()]

    async def aclose(self) -> None:
        await self._client.aclose()
