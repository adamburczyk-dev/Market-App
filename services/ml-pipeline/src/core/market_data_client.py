"""HTTP client for querying historical OHLCV from the market-data service.

Per the architecture rule: events for notifications, HTTP for queries.
Training pulls the universe's history on demand — a plain query client.
"""

from typing import Protocol

import httpx
import structlog
from trading_common.schemas import Interval, OHLCVBar

logger = structlog.get_logger()


class MarketDataClient(Protocol):
    async def get_ohlcv(
        self, symbol: str, interval: Interval, limit: int = 500
    ) -> list[OHLCVBar]: ...

    async def aclose(self) -> None: ...


class HttpMarketDataClient:
    def __init__(self, base_url: str, timeout_s: float = 30.0) -> None:
        self._base = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout_s)

    async def get_ohlcv(self, symbol: str, interval: Interval, limit: int = 500) -> list[OHLCVBar]:
        url = f"{self._base}/api/v1/market-data/ohlcv/{symbol}"
        resp = await self._client.get(url, params={"interval": interval.value, "limit": limit})
        resp.raise_for_status()
        return [OHLCVBar.model_validate(item) for item in resp.json()]

    async def aclose(self) -> None:
        await self._client.aclose()
