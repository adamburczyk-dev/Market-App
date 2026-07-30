"""MarketDataService — orchestrates fetch, storage, cache and event publishing."""

import asyncio
from datetime import UTC, datetime
from typing import Any

import structlog
from trading_common.events import MarketDataUpdatedEvent
from trading_common.schemas import Interval, OHLCVBar

from src.core.cache import Cache
from src.core.fetchers.base import Fetcher
from src.core.incremental import adjustment_drifted, full_history_plan, plan_fetch
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
            # The cache key is (symbol, interval) — it carries no limit — while the
            # cached window was itself produced by SOME limit. Serving a larger
            # request from a shorter window would silently return fewer bars than
            # asked for: feature-engine's limit=250 read used to poison the entry
            # and training then received 250 bars instead of the 2000 it requested.
            # A cached window may only answer requests it actually covers.
            if cached is not None and len(cached) >= limit:
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

    async def sync_symbol(
        self,
        symbol: str,
        interval: Interval,
        now: datetime | None = None,
        overlap_days: int = 5,
        initial_history_days: int = 365 * 6,
    ) -> dict[str, Any]:
        """Bring one symbol up to date, resuming from the newest bar we hold.

        Three outcomes, and the third is the one worth having:

        * nothing stored → full history;
        * stored → fetch from `overlap_days` before the newest bar, so a missed
          run repairs itself instead of leaving a permanent hole;
        * the overlap reveals that adj_close was restated → the symbol's whole
          history is on the wrong scale, so it is re-fetched in full.

        Without the third case an incremental schedule would quietly corrupt the
        adjusted series every time a name split or paid a dividend, and nothing
        downstream would report an error — the numbers would just be wrong.
        """
        moment = now or datetime.now(UTC)
        latest = await self._repository.latest_timestamp(symbol, interval)
        plan = plan_fetch(latest, moment, overlap_days, initial_history_days)

        bars = await self._fetcher.fetch(symbol, interval, plan.start, plan.end)
        mode = plan.mode
        if not plan.is_full and bars:
            stored = await self._repository.get_bars(
                symbol, interval, start=plan.start, end=plan.end, limit=len(bars) + 10
            )
            if adjustment_drifted(stored, bars):
                # Reach back over everything we hold, not just the default
                # depth: the restatement applies to the whole history, so a
                # shorter repair would leave the older part on the old scale.
                earliest = await self._repository.earliest_timestamp(symbol, interval)
                repair = full_history_plan(moment, initial_history_days, earliest)
                bars = await self._fetcher.fetch(symbol, interval, repair.start, repair.end)
                mode = "readjusted"

        count = await self._repository.save_bars(bars)
        await self._cache.invalidate(symbol, interval)
        if count:
            await self._publisher.publish(
                MarketDataUpdatedEvent(symbol=symbol, interval=interval.value, rows_count=count)
            )
        newest = bars[-1].timestamp if bars else latest
        logger.info(
            "Symbol synced",
            symbol=symbol,
            mode=mode,
            rows=count,
            since=latest.isoformat() if latest else None,
        )
        return {
            "symbol": symbol,
            "mode": mode,
            "rows": count,
            "previous_latest": latest.isoformat() if latest else None,
            "latest": newest.isoformat() if newest else None,
        }

    async def sync_universe(
        self,
        symbols: list[str],
        interval: Interval,
        pause_s: float = 0.0,
        now: datetime | None = None,
        overlap_days: int = 5,
        initial_history_days: int = 365 * 6,
    ) -> dict[str, Any]:
        """Sync every symbol, surviving individual failures.

        One unreachable ticker must not cost the other 485 their update — the
        scheduled run is the only thing keeping the event chain alive, so it
        reports failures rather than aborting on them.
        """
        results: list[dict[str, Any]] = []
        failed: dict[str, str] = {}
        for index, symbol in enumerate(symbols):
            try:
                results.append(
                    await self.sync_symbol(
                        symbol, interval, now, overlap_days, initial_history_days
                    )
                )
            except Exception as exc:  # noqa: BLE001 - one bad symbol is not a failed run
                failed[symbol] = f"{type(exc).__name__}: {exc}"
                logger.warning("Symbol sync failed", symbol=symbol, error=str(exc))
            if pause_s and index + 1 < len(symbols):
                await asyncio.sleep(pause_s)

        summary = {
            "symbols": len(symbols),
            "synced": len(results),
            "failed": failed,
            "rows": sum(int(r["rows"]) for r in results),
            "readjusted": [r["symbol"] for r in results if r["mode"] == "readjusted"],
            "results": results,
        }
        logger.info(
            "Universe synced",
            symbols=len(symbols),
            synced=len(results),
            failed=len(failed),
            rows=summary["rows"],
        )
        return summary

    async def latest_timestamp(self, symbol: str, interval: Interval) -> datetime | None:
        return await self._repository.latest_timestamp(symbol, interval)

    async def list_symbols(self) -> list[str]:
        return await self._repository.list_symbols()
