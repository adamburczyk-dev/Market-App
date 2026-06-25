"""SQLAlchemy async ORM and engine helpers for market-data storage.

The ORM table is intentionally bound to no schema. In Postgres the engine sets
``search_path=market_data`` so ``ohlcv`` resolves to ``market_data.ohlcv``
(created as a TimescaleDB hypertable by ``init-db.sql``). In sqlite (tests) the
table simply lives in the default schema.
"""

from datetime import datetime

from sqlalchemy import DateTime, Float, String
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class OHLCVRow(Base):
    """One OHLCV bar. Natural composite PK (symbol, interval, ts)."""

    __tablename__ = "ohlcv"

    symbol: Mapped[str] = mapped_column(String, primary_key=True)
    interval: Mapped[str] = mapped_column(String, primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float)
    source: Mapped[str | None] = mapped_column(String, nullable=True)
    # NOTE: created_at is handled by the DB (server default NOW()) and intentionally
    # not mapped here, so idempotent merges never overwrite it.


def make_engine(database_url: str) -> AsyncEngine:
    """Create an async engine. For Postgres, route unqualified tables to market_data."""
    connect_args: dict = {}
    if database_url.startswith("postgresql"):
        connect_args["server_settings"] = {"search_path": "market_data,public"}
    return create_async_engine(database_url, pool_pre_ping=True, connect_args=connect_args)


def make_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)
