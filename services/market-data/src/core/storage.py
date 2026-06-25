"""Async persistence layer for OHLCV bars."""

from datetime import datetime

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from trading_common.schemas import Interval, OHLCVBar

from src.models.db import OHLCVRow

logger = structlog.get_logger()


def _to_row(bar: OHLCVBar) -> OHLCVRow:
    return OHLCVRow(
        symbol=bar.symbol,
        interval=bar.interval.value,
        ts=bar.timestamp,
        open=bar.open,
        high=bar.high,
        low=bar.low,
        close=bar.close,
        volume=bar.volume,
        source=bar.source,
    )


def _to_bar(row: OHLCVRow) -> OHLCVBar:
    return OHLCVBar(
        symbol=row.symbol,
        timestamp=row.ts,
        interval=Interval(row.interval),
        open=row.open,
        high=row.high,
        low=row.low,
        close=row.close,
        volume=row.volume,
        source=row.source,
    )


class OHLCVRepository:
    """Read/write OHLCV bars. Upserts are idempotent on (symbol, interval, ts)."""

    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession]) -> None:
        self._sessionmaker = sessionmaker

    async def save_bars(self, bars: list[OHLCVBar]) -> int:
        """Idempotently persist bars. Returns number of bars written.

        Uses ``session.merge`` (upsert by primary key) so re-fetching the same
        window does not create duplicates. For very large batches a bulk
        ``ON CONFLICT`` insert would be faster — left as a future optimization.
        """
        if not bars:
            return 0
        async with self._sessionmaker() as session:
            for bar in bars:
                await session.merge(_to_row(bar))
            await session.commit()
        logger.info("Saved OHLCV bars", count=len(bars), symbol=bars[0].symbol)
        return len(bars)

    async def get_bars(
        self,
        symbol: str,
        interval: Interval,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 500,
    ) -> list[OHLCVBar]:
        """Return bars in chronological (oldest-first) order."""
        async with self._sessionmaker() as session:
            stmt = select(OHLCVRow).where(
                OHLCVRow.symbol == symbol,
                OHLCVRow.interval == interval.value,
            )
            if start is not None:
                stmt = stmt.where(OHLCVRow.ts >= start)
            if end is not None:
                stmt = stmt.where(OHLCVRow.ts <= end)
            # Take the most recent `limit` rows, then return chronologically.
            stmt = stmt.order_by(OHLCVRow.ts.desc()).limit(limit)
            result = await session.execute(stmt)
            rows = list(result.scalars().all())
        return [_to_bar(row) for row in reversed(rows)]

    async def list_symbols(self) -> list[str]:
        async with self._sessionmaker() as session:
            stmt = select(OHLCVRow.symbol).distinct().order_by(OHLCVRow.symbol)
            result = await session.execute(stmt)
            return list(result.scalars().all())
