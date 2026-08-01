"""The panel has to be able to hold history, not just the present.

`refresh` keeps the newest two filings — the right answer to "what do we know
now", which is what serving asks. It was also the only path that ever wrote to
the panel, so the point-in-time store could hold at most two years per symbol
and a training join over twenty was impossible by construction. The XBRL
response already carried the whole history; it was being sliced away.
"""

from datetime import UTC, date, datetime

import pytest
from sqlalchemy.ext.asyncio import create_async_engine
from trading_common.events import FundamentalsUpdatedEvent

from src.core.repository import SqlFundamentalsStore
from src.core.service import FundamentalDataService
from src.models.db import Base, make_sessionmaker

from .conftest import FakeFetcher, stmt


@pytest.fixture
async def store(tmp_path):
    """A REAL panel — the whole point here is what reaches persistent history."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/panel.sqlite")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield SqlFundamentalsStore(make_sessionmaker(engine))
    await engine.dispose()


def build_service(fetcher, store, publisher=None):  # type: ignore[no-untyped-def]
    from src.events.publisher import NullPublisher

    return FundamentalDataService(fetcher, publisher or NullPublisher(), store=store)


class RecordingPublisher:
    def __init__(self) -> None:
        self.published: list[FundamentalsUpdatedEvent] = []

    async def publish(self, event) -> None:  # type: ignore[no-untyped-def]
        self.published.append(event)


def twenty_years(symbol: str = "AAPL") -> list:
    """Newest first, exactly as EdgarClient returns them."""
    return [
        stmt(
            date(2026 - i, 12, 31),
            revenue=1000 + 10 * (20 - i),
            net_income=100 + 5 * (20 - i),
            total_assets=1000,
            total_liabilities=400,
            operating_cash_flow=150,
            current_assets=500,
            current_liabilities=200,
            shares_outstanding=100,
            symbol=symbol,
        )
        for i in range(20)
    ]


@pytest.mark.asyncio
async def test_the_whole_history_reaches_the_panel(store):
    service = build_service(FakeFetcher(twenty_years(), enabled=True), store)
    stored = await service.refresh_history("AAPL", periods=24)
    assert stored == 20

    panel = await service.panel(["AAPL"])
    assert len(panel) == 20
    years = sorted(s.period_end.year for s in panel)
    assert years[0] == 2007 and years[-1] == 2026


@pytest.mark.asyncio
async def test_plain_refresh_still_keeps_only_the_present(store):
    """The serving path is unchanged — this is an addition, not a redefinition."""
    service = build_service(FakeFetcher(twenty_years(), enabled=True), store)
    assert await service.refresh("AAPL") is not None
    assert len(await service.panel(["AAPL"])) == 2


@pytest.mark.asyncio
async def test_each_period_is_scored_against_its_own_predecessor(store):
    """An F-Score compares CONSECUTIVE years. Scoring 2012 against 2025 would
    produce a number that looks like a score and means nothing."""
    statements = twenty_years()
    # make ONE year a clear deterioration: a loss, cash burn, more leverage
    bad = next(s for s in statements if s.period_end.year == 2015)
    statements[statements.index(bad)] = bad.model_copy(
        update={"net_income": -200.0, "operating_cash_flow": -50.0, "total_liabilities": 900.0}
    )
    service = build_service(FakeFetcher(statements, enabled=True), store)
    await service.refresh_history("AAPL", periods=24)
    panel = {s.period_end.year: s for s in await service.panel(["AAPL"])}

    # The damage must be LOCAL. Were every period scored against the newest
    # filing instead of its own predecessor, one bad year could not show up as
    # one bad score surrounded by healthy ones.
    assert panel[2015].piotroski_f_score is not None
    assert panel[2015].piotroski_f_score <= 3
    assert panel[2017].piotroski_f_score >= 5
    assert panel[2026].piotroski_f_score >= 5


@pytest.mark.asyncio
async def test_history_publishes_once_not_once_per_year(store):
    """`fundamentals.updated` says current knowledge changed. Replaying twenty
    years of it would wake every downstream consumer twenty times per symbol."""
    publisher = RecordingPublisher()
    service = build_service(FakeFetcher(twenty_years(), enabled=True), store, publisher=publisher)
    await service.refresh_history("AAPL", periods=24)

    assert len(publisher.published) == 1
    assert publisher.published[0].period_end == "2026-12-31"


@pytest.mark.asyncio
async def test_a_symbol_edgar_does_not_know_stores_nothing(store):
    service = build_service(FakeFetcher([], enabled=True), store)
    assert await service.refresh_history("NOPE") == 0
    assert await service.panel(["NOPE"]) == []


@pytest.mark.asyncio
async def test_the_as_of_read_can_now_answer_about_the_past(store):
    """The point of the whole exercise: what was knowable in 2015.

    With only the newest two filings in the panel this read had nothing to
    return, so a training join produced neutral fills for every historical
    session — a feature family present in name only.
    """
    statements = [
        s.model_copy(update={"filed_at": datetime(s.period_end.year + 1, 3, 1, tzinfo=UTC)})
        for s in twenty_years()
    ]
    service = build_service(FakeFetcher(statements, enabled=True), store)
    await service.refresh_history("AAPL", periods=24)

    known_in_2016 = await service.available_before("AAPL", datetime(2016, 6, 1, tzinfo=UTC))
    assert known_in_2016 is not None
    assert known_in_2016.period_end.year == 2015

    # ...and nothing filed later leaks backwards
    known_in_2010 = await service.available_before("AAPL", datetime(2010, 6, 1, tzinfo=UTC))
    assert known_in_2010 is not None
    assert known_in_2010.period_end.year <= 2009


@pytest.mark.asyncio
async def test_backfill_route_reports_how_many_periods_landed(
    wired: tuple[object, FundamentalDataService], store
):
    client, _ = wired
    from src.api.deps import get_service
    from src.main import app

    service = build_service(FakeFetcher(twenty_years(), enabled=True), store)
    app.dependency_overrides[get_service] = lambda: service
    resp = await client.post("/api/v1/fundamental-data/backfill/aapl?periods=24")  # type: ignore[attr-defined]
    assert resp.status_code == 200
    assert resp.json() == {"symbol": "AAPL", "periods": 20}

    empty = build_service(FakeFetcher([], enabled=True), store)
    app.dependency_overrides[get_service] = lambda: empty
    missing = await client.post("/api/v1/fundamental-data/backfill/NOPE")  # type: ignore[attr-defined]
    assert missing.status_code == 404
