"""Storage for macro observations, with point-in-time reads.

The read that matters is two-dimensional and neither axis is optional:

    "For each series, what was the newest published number ON day D,
     using only values that were already public ON day D?"

Getting either half wrong is a different bug. Ignoring `realtime_start` feeds
the model revisions that did not exist yet. Ignoring `observation_date` picks
whatever was revised most recently rather than the most recent period.

Everything degrades to an empty answer rather than a wrong one: a series with
no usable row simply does not appear in the as-of result, and the caller sees a
regime it cannot classify instead of one classified from stale inputs.
"""

from datetime import date
from typing import Protocol

import structlog
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from trading_common.schemas import MacroObservation

from src.models.db import UNKNOWN_VINTAGE, MacroObservationRow

logger = structlog.get_logger()


class MacroStore(Protocol):
    async def save(self, observations: list[MacroObservation]) -> int: ...
    async def as_of(self, day: date) -> dict[str, float]: ...
    async def panel(self) -> list[MacroObservation]: ...
    async def series_history(
        self, series: str, start: date | None = None, end: date | None = None
    ) -> list[MacroObservation]: ...
    async def coverage(self) -> dict[str, dict[str, str | int]]: ...


class NullMacroStore:
    """Used when the database is unavailable — the service still serves live state."""

    async def save(self, observations: list[MacroObservation]) -> int:
        return 0

    async def as_of(self, day: date) -> dict[str, float]:
        return {}

    async def panel(self) -> list[MacroObservation]:
        return []

    async def series_history(
        self, series: str, start: date | None = None, end: date | None = None
    ) -> list[MacroObservation]:
        return []

    async def coverage(self) -> dict[str, dict[str, str | int]]:
        return {}


# Bind-parameter budget. PostgreSQL caps a statement at 65535 parameters and
# SQLite at 999 by default; this row binds five columns, so the smaller ceiling
# governs and is applied to both — a backfill is not hot enough to justify two
# code paths.
COLUMNS_PER_ROW = 5
MAX_ROWS_PER_INSERT = 999 // COLUMNS_PER_ROW


def _deduplicate(observations: list[MacroObservation]) -> list[MacroObservation]:
    """Last occurrence wins per natural key.

    Postgres refuses an `ON CONFLICT DO UPDATE` whose own VALUES list names the
    same key twice ("cannot affect row a second time"), so ONE duplicated row in
    a batch would fail the whole write — the regression bulk upserts caused in
    market-data. Dedupe here, and say how many collapsed.
    """
    unique: dict[tuple[str, date, date], MacroObservation] = {}
    for obs in observations:
        key = (obs.series, obs.observation_date, obs.realtime_start or UNKNOWN_VINTAGE)
        unique[key] = obs
    if len(unique) != len(observations):
        logger.info("Collapsed duplicate observations", dropped=len(observations) - len(unique))
    return list(unique.values())


class SqlMacroStore:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = session_factory

    async def save(self, observations: list[MacroObservation]) -> int:
        """Bulk upsert. Idempotent: re-running a backfill changes nothing."""
        rows = _deduplicate(observations)
        if not rows:
            return 0
        payload = [
            {
                "series": obs.series,
                "observation_date": obs.observation_date,
                "realtime_start": obs.realtime_start or UNKNOWN_VINTAGE,
                "value": obs.value,
                "source": obs.source,
            }
            for obs in rows
        ]
        async with self._sessions() as session:
            dialect = session.bind.dialect.name if session.bind is not None else "postgresql"
            insert = sqlite_insert if dialect == "sqlite" else pg_insert
            # One statement per CHUNK, not per call. A wire protocol carries a
            # bounded number of bind parameters (65535 on PostgreSQL, 999 by
            # default on SQLite), and this table binds COLUMNS_PER_ROW of them
            # per row — so a single daily series over 20 years of vintages
            # blows the limit and takes the whole backfill with it. The failure
            # only appears at real volume: every test fixture fits in one
            # statement.
            for start in range(0, len(payload), MAX_ROWS_PER_INSERT):
                chunk = payload[start : start + MAX_ROWS_PER_INSERT]
                statement = insert(MacroObservationRow).values(chunk)
                statement = statement.on_conflict_do_update(
                    index_elements=["series", "observation_date", "realtime_start"],
                    set_={"value": statement.excluded.value, "source": statement.excluded.source},
                )
                await session.execute(statement)
            await session.commit()
        return len(payload)

    async def as_of(self, day: date) -> dict[str, float]:
        """Series → value that was public on `day`, for the newest period then known.

        One query per series would be simpler; this walks a single ordered scan
        instead because a 20-year backfill asks this question once per SESSION,
        and 5000 sessions × 6 series is 30 000 round trips.
        """
        async with self._sessions() as session:
            result = await session.execute(
                select(
                    MacroObservationRow.series,
                    MacroObservationRow.observation_date,
                    MacroObservationRow.value,
                )
                .where(MacroObservationRow.realtime_start <= day)
                .where(MacroObservationRow.observation_date <= day)
                .order_by(
                    MacroObservationRow.series,
                    MacroObservationRow.observation_date.desc(),
                    MacroObservationRow.realtime_start.desc(),
                )
            )
            latest: dict[str, float] = {}
            for series, _observation_date, value in result.all():
                # Ordered newest-period-first, and within a period newest
                # vintage first, so the first row per series is the answer.
                if series not in latest:
                    latest[series] = float(value)
            return latest

    async def panel(self) -> list[MacroObservation]:
        """Every stored observation, every vintage — for a one-pass history walk.

        Answering "regime on day D" per day would be one query per day: a
        20-year request is 7300 round trips to compute something that fits in
        memory many times over.
        """
        async with self._sessions() as session:
            rows = (
                await session.execute(
                    select(MacroObservationRow).order_by(
                        MacroObservationRow.observation_date,
                        MacroObservationRow.realtime_start,
                    )
                )
            ).scalars()
            return [_to_schema(row) for row in rows]

    async def series_history(
        self, series: str, start: date | None = None, end: date | None = None
    ) -> list[MacroObservation]:
        async with self._sessions() as session:
            query = select(MacroObservationRow).where(MacroObservationRow.series == series)
            if start is not None:
                query = query.where(MacroObservationRow.observation_date >= start)
            if end is not None:
                query = query.where(MacroObservationRow.observation_date <= end)
            rows = (
                await session.execute(
                    query.order_by(
                        MacroObservationRow.observation_date,
                        MacroObservationRow.realtime_start,
                    )
                )
            ).scalars()
            return [_to_schema(row) for row in rows]

    async def coverage(self) -> dict[str, dict[str, str | int]]:
        """Per series: how many rows, over what period, and how many are undated.

        `undated` is reported because those rows are invisible to as-of reads —
        a panel that looks full and answers nothing historical is otherwise
        indistinguishable from one that is genuinely populated.
        """
        async with self._sessions() as session:
            rows = (await session.execute(select(MacroObservationRow))).scalars().all()
        summary: dict[str, dict[str, str | int]] = {}
        for row in rows:
            entry = summary.setdefault(
                row.series, {"rows": 0, "undated": 0, "first": "", "last": ""}
            )
            entry["rows"] = int(entry["rows"]) + 1
            if row.realtime_start == UNKNOWN_VINTAGE:
                entry["undated"] = int(entry["undated"]) + 1
            iso = row.observation_date.isoformat()
            if not entry["first"] or iso < str(entry["first"]):
                entry["first"] = iso
            if not entry["last"] or iso > str(entry["last"]):
                entry["last"] = iso
        return summary


def _to_schema(row: MacroObservationRow) -> MacroObservation:
    return MacroObservation(
        series=row.series,
        observation_date=row.observation_date,
        value=row.value,
        realtime_start=(None if row.realtime_start == UNKNOWN_VINTAGE else row.realtime_start),
        source=row.source,
    )
