"""Testy warstwy storage na in-memory SQLite."""

from datetime import UTC, datetime

import pytest
from trading_common.schemas import Interval

from src.core.storage import OHLCVRepository

from .conftest import make_bar


@pytest.mark.asyncio
async def test_save_and_get_roundtrip(repository: OHLCVRepository):
    bars = [make_bar(close=c, day=d) for d, c in enumerate([100, 101, 102], start=1)]
    written = await repository.save_bars(bars)
    assert written == 3

    fetched = await repository.get_bars("AAPL", Interval.D1)
    assert len(fetched) == 3
    # zwracane chronologicznie (rosnąco po czasie)
    assert [b.close for b in fetched] == [100, 101, 102]


@pytest.mark.asyncio
async def test_save_is_idempotent(repository: OHLCVRepository):
    bar = make_bar(close=100, day=1)
    await repository.save_bars([bar])
    # ten sam (symbol, interval, ts) z inną ceną — upsert, nie duplikat
    await repository.save_bars([make_bar(close=999, day=1)])

    fetched = await repository.get_bars("AAPL", Interval.D1)
    assert len(fetched) == 1
    assert fetched[0].close == 999


@pytest.mark.asyncio
async def test_a_payload_that_repeats_a_bar_does_not_fail_the_whole_symbol(
    repository: OHLCVRepository,
):
    """Postgres refuses an ON CONFLICT DO UPDATE that names one key twice
    ("cannot affect row a second time"), so a provider returning a duplicated
    bar took down the whole fetch with a 500 — and every retry with it, since
    the payload is deterministic. The per-bar merge this replaced tolerated it.
    """
    written = await repository.save_bars(
        [make_bar(close=100, day=1), make_bar(close=101, day=2), make_bar(close=999, day=1)]
    )
    assert written == 2  # the count reported is what was really stored

    fetched = await repository.get_bars("AAPL", Interval.D1)
    assert [b.close for b in fetched] == [999, 101]  # last occurrence wins


@pytest.mark.asyncio
async def test_a_naive_duplicate_of_an_aware_bar_is_also_collapsed(
    repository: OHLCVRepository,
):
    """The pair a naive dedupe would miss: two different Python datetimes, one
    single row in a TIMESTAMPTZ column — so the constraint still fires."""
    aware = make_bar(close=100, day=3)
    naive = make_bar(close=555, day=3)
    naive = naive.model_copy(update={"timestamp": aware.timestamp.replace(tzinfo=None)})

    assert await repository.save_bars([aware, naive]) == 1
    fetched = await repository.get_bars("AAPL", Interval.D1)
    assert [b.close for b in fetched] == [555]


@pytest.mark.asyncio
async def test_empty_save_returns_zero(repository: OHLCVRepository):
    assert await repository.save_bars([]) == 0


@pytest.mark.asyncio
async def test_get_respects_limit_and_returns_latest(repository: OHLCVRepository):
    bars = [make_bar(close=100 + d, day=d) for d in range(1, 6)]
    await repository.save_bars(bars)
    fetched = await repository.get_bars("AAPL", Interval.D1, limit=2)
    # 2 najnowsze, zwrócone chronologicznie
    assert [b.close for b in fetched] == [104, 105]


@pytest.mark.asyncio
async def test_get_filters_by_date_range(repository: OHLCVRepository):
    await repository.save_bars([make_bar(close=100 + d, day=d) for d in range(1, 6)])
    start = datetime(2024, 1, 2, tzinfo=UTC)
    end = datetime(2024, 1, 4, tzinfo=UTC)
    fetched = await repository.get_bars("AAPL", Interval.D1, start=start, end=end)
    assert [b.close for b in fetched] == [102, 103, 104]


@pytest.mark.asyncio
async def test_list_symbols(repository: OHLCVRepository):
    await repository.save_bars([make_bar(symbol="AAPL", day=1), make_bar(symbol="MSFT", day=1)])
    assert await repository.list_symbols() == ["AAPL", "MSFT"]
