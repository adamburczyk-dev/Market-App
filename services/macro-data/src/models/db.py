"""SQLAlchemy async ORM for the macro observation panel — the VINTAGE store.

The service kept one in-memory snapshot: the current regime and nothing else.
That answers "what is the regime now" and no other question, so
`build_dataset`'s `regime_by_date` had nothing to fill it with and the five
`macro_*` columns came out all-zero in every training run — present by name
only, then dropped by the zero-variance filter (P2-4).

Rows are keyed by `(series, observation_date, realtime_start)` because a macro
series has TWO time axes and both are load-bearing:

- `observation_date` — the period the number describes (March 2015).
- `realtime_start`   — the date from which that number was the published value.

The same March-2015 unemployment rate exists many times over: the initial
print, the next month's revision, the annual benchmark revision years later.
Keeping only the newest is exactly what turns a macro feature into look-ahead —
the model would be told what March 2015 turned out to be, not what anyone knew
at the time.
"""

from datetime import date

from sqlalchemy import Date, Float, String
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

SCHEMA = "macro_data"


class Base(DeclarativeBase):
    pass


class MacroObservationRow(Base):
    """One (series, period, vintage) reading. Natural PK → idempotent upserts."""

    __tablename__ = "macro_observations"

    series: Mapped[str] = mapped_column(String, primary_key=True)
    observation_date: Mapped[date] = mapped_column(Date, primary_key=True)
    # Part of the key: the same period legitimately has several values, one per
    # revision. Stored NOT NULL with the sentinel below rather than NULL,
    # because NULL in a primary key is not comparable and such a row would be
    # silently un-upsertable — it would duplicate on every backfill.
    realtime_start: Mapped[date] = mapped_column(Date, primary_key=True)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    source: Mapped[str] = mapped_column(String, nullable=False, default="fred")


#: Stand-in `realtime_start` for a reading fetched WITHOUT vintage information.
#: Far in the future on purpose: an as-of read filters `realtime_start <= day`,
#: so such a row can never satisfy a historical question — it is reachable only
#: by "what do we know now". Encoding "undated" as a very OLD date would have
#: done the opposite and made every undated row look like the earliest thing we
#: ever knew, which is precisely the look-ahead this table exists to prevent.
UNKNOWN_VINTAGE = date(9999, 12, 31)


def make_engine(database_url: str) -> AsyncEngine:
    """Create an async engine. For Postgres, route unqualified tables to macro_data."""
    connect_args: dict = {}
    if database_url.startswith("postgresql"):
        connect_args["server_settings"] = {"search_path": f"{SCHEMA},public"}
    return create_async_engine(database_url, pool_pre_ping=True, connect_args=connect_args)


def make_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)
