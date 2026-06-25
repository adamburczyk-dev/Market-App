"""Fetcher abstraction — every data source returns a list of OHLCVBar."""

from abc import ABC, abstractmethod
from datetime import datetime

import structlog
from trading_common.schemas import Interval, OHLCVBar

logger = structlog.get_logger()


class FetchError(RuntimeError):
    """Raised when a data source fails to return usable data."""


class Fetcher(ABC):
    """Base class for OHLCV data sources."""

    name: str = "base"

    @abstractmethod
    async def fetch(
        self,
        symbol: str,
        interval: Interval,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[OHLCVBar]:
        """Fetch OHLCV bars for ``symbol``. May return an empty list."""
        raise NotImplementedError


class FallbackFetcher(Fetcher):
    """Try each fetcher in order; return the first non-empty result."""

    name = "fallback"

    def __init__(self, fetchers: list[Fetcher]) -> None:
        if not fetchers:
            raise ValueError("FallbackFetcher requires at least one fetcher")
        self._fetchers = fetchers

    async def fetch(
        self,
        symbol: str,
        interval: Interval,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[OHLCVBar]:
        last_error: Exception | None = None
        for fetcher in self._fetchers:
            try:
                bars = await fetcher.fetch(symbol, interval, start, end)
            except Exception as exc:  # noqa: BLE001 - try the next source
                last_error = exc
                logger.warning("Fetcher failed, trying next", source=fetcher.name, error=str(exc))
                continue
            if bars:
                return bars
        if last_error is not None:
            raise FetchError(f"all fetchers failed for {symbol}") from last_error
        return []
