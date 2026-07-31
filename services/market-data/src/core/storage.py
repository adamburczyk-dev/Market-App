"""Async persistence layer for OHLCV bars."""

from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import structlog
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from trading_common.schemas import Interval, OHLCVBar
from trading_common.timeutil import as_utc

from src.models.db import OHLCVRow

logger = structlog.get_logger()

# Everything except the primary key and the DB-owned created_at. A re-fetch
# must be able to correct a bar (providers restate them, and adj_close changes
# retroactively on every split and dividend).
_MUTABLE_COLUMNS = ("open", "high", "low", "close", "volume", "adj_close", "source")
# Postgres caps a statement at 65535 bind parameters; 10 columns per row leaves
# plenty of headroom at this size while keeping the round-trip count low.
_UPSERT_CHUNK = 2_000


def _chunks(bars: list[OHLCVBar], size: int) -> Iterator[list[OHLCVBar]]:
    for start in range(0, len(bars), size):
        yield bars[start : start + size]


def _deduplicate(bars: list[OHLCVBar]) -> list[OHLCVBar]:
    """One bar per (symbol, interval, instant); the LAST occurrence wins.

    Postgres refuses an ``ON CONFLICT DO UPDATE`` whose own VALUES list names
    the same key twice ("cannot affect row a second time"), so a provider that
    returns a bar twice — yfinance does, around repaired rows and partial
    sessions — used to fail the ENTIRE symbol with a 500. The per-bar
    ``session.merge`` this replaced simply applied the second one, so tolerating
    a duplicate payload is not a new leniency; it is behaviour that came back.

    Keyed on the instant, not on the datetime object: a naive and an aware
    timestamp for the same moment are two different Python values and one
    single row in a TIMESTAMPTZ column, which is precisely the pair that would
    slip past a naive dedupe and hit the constraint anyway.
    """
    by_key: dict[tuple[str, str, datetime], OHLCVBar] = {}
    for bar in bars:
        by_key[(bar.symbol, bar.interval.value, as_utc(bar.timestamp).astimezone(UTC))] = bar
    return list(by_key.values())


def _to_values(bar: OHLCVBar) -> dict[str, Any]:
    return {
        "symbol": bar.symbol,
        "interval": bar.interval.value,
        "ts": bar.timestamp,
        "open": bar.open,
        "high": bar.high,
        "low": bar.low,
        "close": bar.close,
        "volume": bar.volume,
        "adj_close": bar.adj_close,
        "source": bar.source,
    }


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
        adj_close=bar.adj_close,
        source=bar.source,
    )


def _to_bar(row: OHLCVRow) -> OHLCVBar:
    return OHLCVBar(
        symbol=row.symbol,
        # Aware on the way out, whatever the backend. sqlite hands back naive
        # timestamps, and a naive/aware mismatch here does not raise — it makes
        # dict lookups by timestamp silently miss, which is how the corporate-
        # action check first reported "no drift" on a history that had been
        # restated.
        timestamp=as_utc(row.ts),
        interval=Interval(row.interval),
        open=row.open,
        high=row.high,
        low=row.low,
        close=row.close,
        volume=row.volume,
        adj_close=row.adj_close,
        source=row.source,
    )


class OHLCVRepository:
    """Read/write OHLCV bars. Upserts are idempotent on (symbol, interval, ts)."""

    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession]) -> None:
        self._sessionmaker = sessionmaker

    async def save_bars(self, bars: list[OHLCVBar]) -> int:
        """Idempotently persist bars. Returns number of bars written.

        Bulk ``ON CONFLICT DO UPDATE`` on the natural key (symbol, interval, ts),
        chunked. The previous implementation called ``session.merge`` once per
        bar, and merge issues a SELECT before deciding to INSERT or UPDATE — so
        a 486-symbol, 20-year backfill meant roughly 2.4 million round trips.

        ``created_at`` is deliberately absent from both the insert and the
        update: the database owns it (server default), and re-fetching a window
        must not rewrite when a bar was first seen.
        """
        if not bars:
            return 0
        unique = _deduplicate(bars)
        if len(unique) != len(bars):
            # Worth a line: a provider repeating bars is a data-quality signal,
            # and silently absorbing it would hide that as effectively as the
            # crash did.
            logger.warning(
                "Duplicate bars in one payload — keeping the last of each",
                symbol=bars[0].symbol,
                received=len(bars),
                unique=len(unique),
            )
        async with self._sessionmaker() as session:
            dialect = session.bind.dialect.name if session.bind is not None else ""
            insert = pg_insert if dialect == "postgresql" else sqlite_insert
            for chunk in _chunks(unique, _UPSERT_CHUNK):
                statement = insert(OHLCVRow).values([_to_values(bar) for bar in chunk])
                await session.execute(
                    statement.on_conflict_do_update(
                        index_elements=["symbol", "interval", "ts"],
                        set_={
                            column: getattr(statement.excluded, column)
                            for column in _MUTABLE_COLUMNS
                        },
                    )
                )
            await session.commit()
        logger.info("Saved OHLCV bars", count=len(unique), symbol=bars[0].symbol)
        return len(unique)

    async def earliest_timestamp(self, symbol: str, interval: Interval) -> datetime | None:
        """Oldest stored bar — how far back a repair has to reach.

        A corporate action restates the WHOLE history, so re-fetching a default
        window would leave everything older than it on the pre-event scale. With
        a 20-year backfill and a 6-year default that is 14 silent years.
        """
        async with self._sessionmaker() as session:
            result = await session.execute(
                select(func.min(OHLCVRow.ts)).where(
                    OHLCVRow.symbol == symbol,
                    OHLCVRow.interval == interval.value,
                )
            )
            earliest = result.scalar_one_or_none()
        return as_utc(earliest) if earliest is not None else None

    async def latest_timestamp(self, symbol: str, interval: Interval) -> datetime | None:
        """Newest stored bar for a symbol, or None when we hold nothing.

        This is what makes the scheduled pull resume from where it stopped
        rather than assuming it ran yesterday — after a week of downtime the
        gap is a week, and the service has to be able to see that.
        """
        async with self._sessionmaker() as session:
            result = await session.execute(
                select(func.max(OHLCVRow.ts)).where(
                    OHLCVRow.symbol == symbol,
                    OHLCVRow.interval == interval.value,
                )
            )
            latest = result.scalar_one_or_none()
        # Postgres TIMESTAMPTZ returns aware, sqlite returns naive — comparing
        # the two raises TypeError, so the caller would work on one backend and
        # crash on the other. Normalize where the difference originates.
        return as_utc(latest) if latest is not None else None

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
