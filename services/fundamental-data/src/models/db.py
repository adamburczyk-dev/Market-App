"""SQLAlchemy async ORM for the fundamentals PANEL.

The service kept latest-per-symbol in memory, which answers "what do we know
now" and nothing else. Training needs "what was known on 2022-03-14", and the
difference is the whole point of P2-3: joining today's F-score onto a 2022
session teaches the model facts that were published two years later.

Rows are therefore keyed by (symbol, period_end, fiscal_period) and carry
``filed_at`` — the date the statement became knowable. An as-of read takes the
latest row whose ``filed_at`` is on or before the session, and rows without a
``filed_at`` are invisible to it: a fact we cannot date cannot be used
point-in-time.
"""

from datetime import date, datetime

from sqlalchemy import Date, DateTime, Float, Integer, String
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class FundamentalsRow(Base):
    """One reporting period for one symbol. Natural PK; upserts are idempotent."""

    __tablename__ = "fundamentals"

    symbol: Mapped[str] = mapped_column(String, primary_key=True)
    period_end: Mapped[date] = mapped_column(Date, primary_key=True)
    fiscal_period: Mapped[str] = mapped_column(String, primary_key=True)
    # NULL means "we do not know when this was published" — such a row is
    # deliberately excluded from as-of reads rather than assumed to be old.
    filed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revenue: Mapped[float | None] = mapped_column(Float, nullable=True)
    net_income: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_assets: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_liabilities: Mapped[float | None] = mapped_column(Float, nullable=True)
    current_assets: Mapped[float | None] = mapped_column(Float, nullable=True)
    current_liabilities: Mapped[float | None] = mapped_column(Float, nullable=True)
    shares_outstanding: Mapped[float | None] = mapped_column(Float, nullable=True)
    operating_cash_flow: Mapped[float | None] = mapped_column(Float, nullable=True)
    eps: Mapped[float | None] = mapped_column(Float, nullable=True)
    piotroski_f_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source: Mapped[str | None] = mapped_column(String, nullable=True)


def make_engine(database_url: str) -> AsyncEngine:
    """Create an async engine. For Postgres, route unqualified tables to market_data."""
    connect_args: dict = {}
    if database_url.startswith("postgresql"):
        connect_args["server_settings"] = {"search_path": "market_data,public"}
    return create_async_engine(database_url, pool_pre_ping=True, connect_args=connect_args)


def make_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)
