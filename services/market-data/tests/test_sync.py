"""Scheduled incremental sync — orchestration against a real sqlite store.

`test_incremental.py` pins the decisions in isolation; this pins that the
service actually acts on them: that the second run asks the provider for a
short window rather than the whole history, that a restated adj_close triggers
a repair, and that one broken symbol does not cost the rest of the universe
its update.
"""

from datetime import UTC, datetime, timedelta

import pytest
from trading_common.schemas import Interval, OHLCVBar

from src.core.cache import InMemoryCache
from src.core.service import MarketDataService
from src.core.storage import OHLCVRepository
from src.events.publisher import NullPublisher

from .conftest import FakeFetcher

NOW = datetime(2026, 7, 30, 23, 0, tzinfo=UTC)


def bar(day: datetime, symbol: str = "AAPL", close: float = 100.0, adj: float = 100.0) -> OHLCVBar:
    return OHLCVBar(
        symbol=symbol,
        timestamp=day,
        interval=Interval.D1,
        open=close,
        high=close * 1.01,
        low=close * 0.99,
        close=close,
        adj_close=adj,
        volume=1_000_000.0,
        source="test",
    )


def history(n: int, symbol: str = "AAPL", adj_ratio: float = 1.0) -> list[OHLCVBar]:
    return [
        bar(NOW - timedelta(days=n - i), symbol, close=100.0 + i, adj=(100.0 + i) * adj_ratio)
        for i in range(n)
    ]


def build(fetcher: FakeFetcher, repository: OHLCVRepository) -> MarketDataService:
    return MarketDataService(fetcher, repository, InMemoryCache(), NullPublisher())


@pytest.mark.asyncio
async def test_the_first_sync_takes_a_full_history(repository: OHLCVRepository):
    fetcher = FakeFetcher(history(5))
    result = await build(fetcher, repository).sync_symbol("AAPL", Interval.D1, now=NOW)
    assert result["mode"] == "full"
    assert result["previous_latest"] is None
    assert result["rows"] == 5
    _, _, start, _ = fetcher.calls[0]
    assert (NOW - start).days > 300, "a first sync must not ask for a short window"


@pytest.mark.asyncio
async def test_the_second_sync_only_asks_for_the_gap(repository: OHLCVRepository):
    """The point of the whole exercise: having stored five years, the next run
    must not drag five years again."""
    await build(FakeFetcher(history(10)), repository).sync_symbol("AAPL", Interval.D1, now=NOW)

    fetcher = FakeFetcher(history(2))
    result = await build(fetcher, repository).sync_symbol(
        "AAPL", Interval.D1, now=NOW, overlap_days=5
    )
    assert result["mode"] == "incremental"
    assert result["previous_latest"] is not None
    _, _, start, _ = fetcher.calls[0]
    assert 5 <= (NOW - start).days <= 7, "expected the newest bar minus the overlap"


@pytest.mark.asyncio
async def test_a_week_of_downtime_is_repaired_not_lost(repository: OHLCVRepository):
    """A day the scheduler missed is gone forever unless the window reaches
    back for it — nothing else in the system ever revisits a past session."""
    stale = [bar(NOW - timedelta(days=n)) for n in (14, 13, 12)]
    await build(FakeFetcher(stale), repository).sync_symbol("AAPL", Interval.D1, now=NOW)

    fetcher = FakeFetcher([bar(NOW - timedelta(days=n)) for n in range(1, 12)])
    await build(fetcher, repository).sync_symbol("AAPL", Interval.D1, now=NOW, overlap_days=5)
    _, _, start, _ = fetcher.calls[0]
    assert (NOW - start).days >= 12, "the window did not reach back over the outage"


@pytest.mark.asyncio
async def test_a_restated_adj_close_triggers_a_full_refetch(repository: OHLCVRepository):
    """The silent one. A split rewrites adj_close for the whole history; if the
    sync only appended, old bars would keep the pre-split scale and every
    return spanning the split would be wrong — with no error anywhere."""
    await build(FakeFetcher(history(10)), repository).sync_symbol("AAPL", Interval.D1, now=NOW)

    # provider now reports every bar at half the adjusted price (2:1 split)
    fetcher = FakeFetcher(history(10, adj_ratio=0.5))
    result = await build(fetcher, repository).sync_symbol(
        "AAPL", Interval.D1, now=NOW, overlap_days=5
    )
    assert result["mode"] == "readjusted"
    assert len(fetcher.calls) == 2, "expected the incremental probe, then a full repair"
    _, _, repair_start, _ = fetcher.calls[1]
    assert (NOW - repair_start).days > 300

    stored = await repository.get_bars("AAPL", Interval.D1, limit=50)
    assert all(b.adj_close is not None and b.adj_close < b.close for b in stored), (
        "the stored history was not rewritten onto the new scale"
    )


@pytest.mark.asyncio
async def test_an_unchanged_history_is_not_refetched(repository: OHLCVRepository):
    """The check has to be quiet in the normal case, or every nightly run drags
    the full history for the whole universe."""
    await build(FakeFetcher(history(10)), repository).sync_symbol("AAPL", Interval.D1, now=NOW)
    fetcher = FakeFetcher(history(10))
    result = await build(fetcher, repository).sync_symbol("AAPL", Interval.D1, now=NOW)
    assert result["mode"] == "incremental"
    assert len(fetcher.calls) == 1


@pytest.mark.asyncio
async def test_one_broken_symbol_does_not_cost_the_others_their_update(
    repository: OHLCVRepository,
):
    class PartlyBroken(FakeFetcher):
        async def fetch(self, symbol, interval, start=None, end=None):  # type: ignore[no-untyped-def]
            if symbol == "BROKEN":
                raise RuntimeError("provider said no")
            return await super().fetch(symbol, interval, start, end)

    fetcher = PartlyBroken(history(3))
    summary = await build(fetcher, repository).sync_universe(
        ["AAPL", "BROKEN", "MSFT"], Interval.D1, now=NOW
    )
    assert summary["synced"] == 2
    assert "BROKEN" in summary["failed"]
    assert "provider said no" in summary["failed"]["BROKEN"]
    assert summary["rows"] > 0


@pytest.mark.asyncio
async def test_the_sync_publishes_so_the_chain_downstream_wakes_up(
    repository: OHLCVRepository,
):
    """market_data.updated is what drives feature-engine, strategy, the
    aggregator, risk-mgmt and execution. A sync that stored bars silently would
    leave the rest of the system asleep."""
    publisher = NullPublisher()
    service = MarketDataService(FakeFetcher(history(3)), repository, InMemoryCache(), publisher)
    await service.sync_symbol("AAPL", Interval.D1, now=NOW)
    assert len(publisher.published) == 1
    assert publisher.published[0].symbol == "AAPL"  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_nothing_new_publishes_nothing(repository: OHLCVRepository):
    publisher = NullPublisher()
    service = MarketDataService(FakeFetcher([]), repository, InMemoryCache(), publisher)
    result = await service.sync_symbol("AAPL", Interval.D1, now=NOW)
    assert result["rows"] == 0
    assert publisher.published == []


@pytest.mark.asyncio
async def test_latest_timestamp_reports_the_newest_stored_bar(repository: OHLCVRepository):
    assert await repository.latest_timestamp("AAPL", Interval.D1) is None
    await repository.save_bars(history(4))
    latest = await repository.latest_timestamp("AAPL", Interval.D1)
    assert latest == NOW - timedelta(days=1)
    # ...and it is per symbol, not a global maximum
    assert await repository.latest_timestamp("NVDA", Interval.D1) is None


@pytest.mark.asyncio
async def test_a_repair_covers_the_whole_stored_history_not_the_default_depth(
    repository: OHLCVRepository,
):
    """Caught on a real Postgres run: repairing a 5000-bar history with a
    4000-bar window left 1000 bars on the pre-split scale. The restatement
    applies to everything we hold, so the repair window has to as well — with a
    20-year backfill and a 6-year default that gap is fourteen silent years.
    """
    deep = [
        bar(NOW - timedelta(days=4000 - i), close=100.0 + i, adj=100.0 + i)
        for i in range(0, 4000, 50)
    ]
    await repository.save_bars(deep)

    # the probe must restate a bar we actually hold, or there is nothing to
    # compare — the newest stored bar is the natural one
    newest = deep[-1]
    fetcher = FakeFetcher([bar(newest.timestamp, close=newest.close, adj=newest.close / 2)])
    service = build(fetcher, repository)
    # a default depth far shorter than what is stored
    await service.sync_symbol("AAPL", Interval.D1, now=NOW, initial_history_days=365)

    assert len(fetcher.calls) == 2, "expected the probe then the repair"
    _, _, repair_start, _ = fetcher.calls[1]
    earliest = await repository.earliest_timestamp("AAPL", Interval.D1)
    assert earliest is not None
    assert repair_start <= earliest, (
        f"repair started {repair_start} but the history reaches back to {earliest}"
    )


@pytest.mark.asyncio
async def test_earliest_timestamp_reports_the_oldest_stored_bar(repository: OHLCVRepository):
    assert await repository.earliest_timestamp("AAPL", Interval.D1) is None
    await repository.save_bars(history(4))
    assert await repository.earliest_timestamp("AAPL", Interval.D1) == NOW - timedelta(days=4)
