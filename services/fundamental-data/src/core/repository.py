"""Persistence for the fundamentals panel — including the as-of read.

`as_of` is the whole reason this layer exists. Every other read ("what is the
latest") is a convenience; the point-in-time read is what makes fundamentals
usable in training without teaching the model the future.
"""

from datetime import datetime
from typing import Protocol

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from trading_common.fundamentals import as_utc
from trading_common.schemas import FinancialStatements

from src.models.db import FundamentalsRow

logger = structlog.get_logger()

_FIELDS = (
    "revenue",
    "gross_profit",
    "cost_of_revenue",
    "net_income",
    "total_assets",
    "total_liabilities",
    "current_assets",
    "current_liabilities",
    "shares_outstanding",
    "operating_cash_flow",
    "eps",
    "piotroski_f_score",
    "source",
)


def _to_row(statement: FinancialStatements) -> FundamentalsRow:
    return FundamentalsRow(
        symbol=statement.symbol.upper(),
        period_end=statement.period_end,
        fiscal_period=statement.fiscal_period,
        filed_at=statement.filed_at,
        **{name: getattr(statement, name) for name in _FIELDS},
    )


def _to_statement(row: FundamentalsRow) -> FinancialStatements:
    return FinancialStatements(
        symbol=row.symbol,
        period_end=row.period_end,
        fiscal_period=row.fiscal_period,
        # sqlite drops the zone on read where Postgres keeps it; normalize here so
        # a caller comparing against a UTC cutoff cannot hit naive-vs-aware.
        filed_at=as_utc(row.filed_at) if row.filed_at is not None else None,
        **{name: getattr(row, name) for name in _FIELDS},
    )


class FundamentalsStore(Protocol):
    async def save(self, statements: list[FinancialStatements]) -> int: ...

    async def available_before(
        self, symbol: str, cutoff: datetime
    ) -> FinancialStatements | None: ...

    async def panel(self, symbols: list[str]) -> list[FinancialStatements]: ...

    async def ping(self) -> bool: ...


class NullFundamentalsStore:
    """No database configured. The service still scores and publishes; it just
    keeps no history, and `/ready` reports that so the degradation is VISIBLE
    rather than silently producing an empty panel at training time."""

    async def save(self, statements: list[FinancialStatements]) -> int:
        return 0

    async def available_before(self, symbol: str, cutoff: datetime) -> FinancialStatements | None:
        return None

    async def panel(self, symbols: list[str]) -> list[FinancialStatements]:
        return []

    async def ping(self) -> bool:
        return False


class SqlFundamentalsStore:
    """Read/write the panel. Upserts are idempotent on the natural PK."""

    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession]) -> None:
        self._sessionmaker = sessionmaker

    async def save(self, statements: list[FinancialStatements]) -> int:
        if not statements:
            return 0
        async with self._sessionmaker() as session:
            for statement in statements:
                await session.merge(_to_row(statement))
            await session.commit()
        return len(statements)

    async def available_before(self, symbol: str, cutoff: datetime) -> FinancialStatements | None:
        """The most recent statement PUBLISHED strictly before `cutoff`.

        Ordered by filed_at, not by period_end: a later fiscal period filed
        after the cutoff is not knowledge we had. Rows with a NULL filed_at are
        excluded by the WHERE clause — undated is unusable, and defaulting them
        in would reintroduce exactly the look-ahead this method exists to stop.
        The SQL mirrors `trading_common.fundamentals.latest_available_before`,
        which is the same rule for callers holding the panel in memory.
        """
        async with self._sessionmaker() as session:
            stmt = (
                select(FundamentalsRow)
                .where(FundamentalsRow.symbol == symbol.upper())
                .where(FundamentalsRow.filed_at.is_not(None))
                .where(FundamentalsRow.filed_at < cutoff)
                .order_by(FundamentalsRow.filed_at.desc(), FundamentalsRow.period_end.desc())
                .limit(1)
            )
            row = (await session.execute(stmt)).scalars().first()
        return _to_statement(row) if row is not None else None

    async def panel(self, symbols: list[str]) -> list[FinancialStatements]:
        """Every stored period for the requested symbols, oldest filing first.

        Training pulls the whole panel once and does the as-of join locally —
        one request instead of symbols x sessions round trips.
        """
        if not symbols:
            return []
        async with self._sessionmaker() as session:
            stmt = (
                select(FundamentalsRow)
                .where(FundamentalsRow.symbol.in_([s.upper() for s in symbols]))
                .order_by(FundamentalsRow.symbol, FundamentalsRow.period_end)
            )
            rows = (await session.execute(stmt)).scalars().all()
        return [_to_statement(row) for row in rows]

    async def ping(self) -> bool:
        try:
            async with self._sessionmaker() as session:
                await session.execute(select(FundamentalsRow).limit(1))
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Fundamentals store unreachable", error=str(exc))
            return False


__all__ = [
    "FundamentalsStore",
    "NullFundamentalsStore",
    "SqlFundamentalsStore",
]
