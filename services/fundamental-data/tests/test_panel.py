"""The fundamentals PANEL and its point-in-time read (P2-3), on a real sqlite DB.

The panel is what turns "what do we know now" into "what was known on date D".
Without it, joining fundamentals into training is look-ahead — and look-ahead
does not fail loudly, it makes the backtest look better.
"""

from datetime import UTC, date, datetime

import pytest
from sqlalchemy.ext.asyncio import create_async_engine
from trading_common.fundamentals import session_cutoff
from trading_common.schemas import FinancialStatements

from src.core.repository import NullFundamentalsStore, SqlFundamentalsStore
from src.core.service import FundamentalDataService
from src.events.publisher import NullPublisher
from src.models.db import Base, make_sessionmaker


def statement(
    symbol: str,
    period_end: str,
    filed_at: str | None,
    *,
    revenue: float = 1000.0,
    net_income: float = 100.0,
    f_score: int | None = None,
) -> FinancialStatements:
    return FinancialStatements(
        symbol=symbol,
        period_end=date.fromisoformat(period_end),
        fiscal_period="FY",
        filed_at=datetime.fromisoformat(filed_at).replace(tzinfo=UTC) if filed_at else None,
        revenue=revenue,
        net_income=net_income,
        total_assets=2000.0,
        total_liabilities=800.0,
        current_assets=900.0,
        current_liabilities=500.0,
        shares_outstanding=100.0,
        operating_cash_flow=150.0,
        eps=1.0,
        piotroski_f_score=f_score,
        source="test",
    )


@pytest.fixture
async def store(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/panel.sqlite")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield SqlFundamentalsStore(make_sessionmaker(engine))
    await engine.dispose()


class StubFetcher:
    enabled = True

    async def latest_statements(self, symbol: str, count: int = 2):
        return []

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_round_trip_and_idempotent_upsert(store):
    first = statement("AAPL", "2023-12-31", "2024-02-02")
    assert await store.save([first]) == 1
    # re-ingesting the same period must not duplicate it, and must take the new value
    await store.save([statement("AAPL", "2023-12-31", "2024-02-02", revenue=1234.0)])
    panel = await store.panel(["AAPL"])
    assert len(panel) == 1
    assert panel[0].revenue == 1234.0
    assert panel[0].filed_at == datetime(2024, 2, 2, tzinfo=UTC)


@pytest.mark.asyncio
async def test_as_of_returns_what_was_published_then(store):
    await store.save(
        [
            statement("AAPL", "2022-12-31", "2023-02-03", f_score=4),
            statement("AAPL", "2023-12-31", "2024-02-02", f_score=7),
            statement("AAPL", "2024-12-31", "2025-02-07", f_score=9),
        ]
    )
    mid_2024 = await store.available_before("AAPL", session_cutoff(date(2024, 6, 14)))
    assert mid_2024 is not None
    assert mid_2024.period_end == date(2023, 12, 31)
    assert mid_2024.piotroski_f_score == 7  # NOT the 2024 filing's 9

    early = await store.available_before("AAPL", session_cutoff(date(2021, 1, 4)))
    assert early is None  # nothing had been published yet


@pytest.mark.asyncio
async def test_the_filing_day_itself_does_not_count(store):
    await store.save([statement("AAPL", "2023-12-31", "2024-02-02")])
    assert await store.available_before("AAPL", session_cutoff(date(2024, 2, 2))) is None
    assert await store.available_before("AAPL", session_cutoff(date(2024, 2, 5))) is not None


@pytest.mark.asyncio
async def test_undated_rows_are_invisible_to_the_point_in_time_read(store):
    """Undated is not old. If an undated row could win the join it would be
    'known' for every session in history, which is the worst case of all."""
    await store.save([statement("AAPL", "2021-12-31", None)])
    assert await store.available_before("AAPL", session_cutoff(date(2030, 1, 1))) is None
    panel = await store.panel(["AAPL"])
    assert len(panel) == 1 and panel[0].filed_at is None  # ...but still stored


@pytest.mark.asyncio
async def test_panel_is_scoped_to_the_symbols_asked_for(store):
    await store.save(
        [
            statement("AAPL", "2023-12-31", "2024-02-02"),
            statement("MSFT", "2023-06-30", "2023-07-27"),
        ]
    )
    assert {s.symbol for s in await store.panel(["AAPL"])} == {"AAPL"}
    assert len(await store.panel(["AAPL", "MSFT"])) == 2
    assert await store.panel([]) == []


@pytest.mark.asyncio
async def test_service_persists_both_periods_it_scored(store):
    """The prior year is a panel row in its own right — a panel that only ever
    holds the newest filing cannot answer an as-of question about last year."""
    service = FundamentalDataService(StubFetcher(), NullPublisher(), store=store)
    await service.ingest(
        current=statement("AAPL", "2023-12-31", "2024-02-02"),
        prior=statement("AAPL", "2022-12-31", "2023-02-03"),
    )
    panel = await service.panel(["AAPL"])
    assert [s.period_end for s in panel] == [date(2022, 12, 31), date(2023, 12, 31)]
    # the scored F-score is what got stored, not a None
    latest = await service.as_of_session("AAPL", date(2024, 6, 14))
    assert latest is not None and latest.piotroski_f_score is not None


@pytest.mark.asyncio
async def test_without_a_database_the_service_still_works_but_has_no_history():
    """Degradation must be visible, not silent: scoring and publishing continue,
    the panel is empty, and `/ready` reports panel=false."""
    service = FundamentalDataService(StubFetcher(), NullPublisher(), store=NullFundamentalsStore())
    scored, breakdown = await service.ingest(current=statement("AAPL", "2023-12-31", "2024-02-02"))
    assert scored.piotroski_f_score == breakdown.score
    assert await service.panel(["AAPL"]) == []
    assert await service.as_of_session("AAPL", date(2024, 6, 14)) is None
    assert await service.store_ready() is False
