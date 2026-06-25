"""MarketDataService — orchestrates fetch, storage, cache and event publishing."""

from datetime import datetime

import structlog
from trading_common.events import MarketDataUpdatedEvent
from trading_common.schemas import Interval, OHLCVBar

from src.core.cache import Cache
from src.core.fetchers.base import Fetcher
from src.core.storage import OHLCVRepository
from src.events.publisher import Publisher

logger = structlog.get_logger()


class MarketDataService:
    def __init__(
        self,
        fetcher: Fetcher,
        repository: OHLCVRepository,
        cache: Cache,
        publisher: Publisher,
    ) -> None:
        self._fetcher = fetcher
        self._repository = repository
        self._cache = cache
        self._publisher = publisher

    async def get_ohlcv(
        self,
        symbol: str,
        interval: Interval,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 500,
    ) -> list[OHLCVBar]:
        """Read bars. The unbounded 'latest' query is cached; ranged queries are not."""
        cacheable = start is None and end is None
        if cacheable:
            cached = await self._cache.get_bars(symbol, interval)
            if cached is not None:
                return cached[-limit:]

        bars = await self._repository.get_bars(symbol, interval, start, end, limit)

        if cacheable and bars:
            await self._cache.set_bars(symbol, interval, bars)
        return bars

    async def fetch_and_store(
        self,
        symbol: str,
        interval: Interval,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> int:
        """Fetch from the data source, persist, invalidate cache, publish event."""
        bars = await self._fetcher.fetch(symbol, interval, start, end)
        count = await self._repository.save_bars(bars)
        await self._cache.invalidate(symbol, interval)
        if count:
            await self._publisher.publish(
                MarketDataUpdatedEvent(symbol=symbol, interval=interval.value, rows_count=count)
            )
        logger.info("Fetch-and-store complete", symbol=symbol, interval=interval.value, rows=count)
        return count

    async def list_symbols(self) -> list[str]:
        return await self._repository.list_symbols()
