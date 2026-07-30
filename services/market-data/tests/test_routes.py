"""Testy HTTP endpointów z w pełni podpiętym serwisem."""

import pytest
from httpx import AsyncClient
from trading_common.constants import MAX_OHLCV_LIMIT

from src.core.service import MarketDataService


@pytest.mark.asyncio
async def test_fetch_then_get_ohlcv(wired: tuple[AsyncClient, MarketDataService]):
    client, _ = wired

    resp = await client.post("/api/v1/market-data/fetch/aapl")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["symbol"] == "AAPL"  # symbol jest normalizowany do uppercase
    assert body["rows"] == 3

    resp = await client.get("/api/v1/market-data/ohlcv/AAPL")
    assert resp.status_code == 200
    bars = resp.json()
    assert len(bars) == 3
    assert bars[0]["symbol"] == "AAPL"


@pytest.mark.asyncio
async def test_get_ohlcv_empty_before_fetch(wired: tuple[AsyncClient, MarketDataService]):
    client, _ = wired
    resp = await client.get("/api/v1/market-data/ohlcv/MSFT")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_symbols_reflects_stored_data(wired: tuple[AsyncClient, MarketDataService]):
    client, _ = wired
    await client.post("/api/v1/market-data/fetch/AAPL")
    resp = await client.get("/api/v1/market-data/symbols")
    assert resp.status_code == 200
    assert resp.json()["symbols"] == ["AAPL"]


@pytest.mark.asyncio
async def test_ohlcv_limit_validation(wired: tuple[AsyncClient, MarketDataService]):
    client, _ = wired
    resp = await client.get("/api/v1/market-data/ohlcv/AAPL", params={"limit": 0})
    assert resp.status_code == 422  # limit musi być >= 1


@pytest.mark.asyncio
async def test_a_twenty_year_window_is_servable(wired: tuple[AsyncClient, MarketDataService]):
    """The read ceiling must cover the longest window anyone actually asks for.

    It did not. ml-pipeline accepted limits up to 10_000 while this route
    stopped at 5_000, so a 20-year training request (5040 sessions + 253
    warm-up bars = 5293) was rejected here with a 422 — and only after the
    campaign had backfilled 455 symbols. The number below is that request.
    """
    client, _ = wired
    resp = await client.get("/api/v1/market-data/ohlcv/AAPL", params={"limit": 20 * 252 + 253})
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_the_read_ceiling_is_the_shared_one(wired: tuple[AsyncClient, MarketDataService]):
    """Both sides of the ceiling, pinned to the shared constant.

    A local literal here is exactly how the mismatch happened: this route is
    the SERVER side of a limit that callers declare for themselves.
    """
    client, _ = wired
    resp = await client.get("/api/v1/market-data/ohlcv/AAPL", params={"limit": MAX_OHLCV_LIMIT})
    assert resp.status_code == 200
    resp = await client.get("/api/v1/market-data/ohlcv/AAPL", params={"limit": MAX_OHLCV_LIMIT + 1})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_unexpected_failure_names_its_cause(wired: tuple[AsyncClient, MarketDataService]):
    """A storage/cache/event failure must not surface as a bare 500.

    Starlette answers unhandled exceptions with an EMPTY plain-text body, so
    the caller (e.g. the bootstrap script) printed "HTTP 500:" and nothing
    else — undiagnosable from the outside. The route names the cause instead.
    """
    client, service = wired

    async def boom(*args, **kwargs):
        raise RuntimeError('relation "ohlcv" does not exist')

    service.fetch_and_store = boom  # type: ignore[method-assign]

    resp = await client.post("/api/v1/market-data/fetch/AAPL")
    assert resp.status_code == 500
    detail = resp.json()["detail"]
    assert "RuntimeError" in detail
    assert 'relation "ohlcv" does not exist' in detail


@pytest.mark.asyncio
async def test_sync_endpoint_runs_the_incremental_pull(
    wired: tuple[AsyncClient, MarketDataService],
):
    client, _ = wired
    resp = await client.post("/api/v1/market-data/sync?symbols=AAPL,MSFT")
    assert resp.status_code == 200
    body = resp.json()
    assert body["symbols"] == 2
    assert body["synced"] == 2
    assert body["rows"] > 0


@pytest.mark.asyncio
async def test_sync_without_symbols_refuses_rather_than_inventing_a_universe(
    wired: tuple[AsyncClient, MarketDataService],
):
    """FETCH_SYMBOLS is empty in tests. Falling back to a default list here
    would have the scheduler quietly pulling names nobody asked for."""
    client, _ = wired
    resp = await client.post("/api/v1/market-data/sync")
    assert resp.status_code == 400
    assert "FETCH_SYMBOLS" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_sync_status_reports_staleness_per_symbol(
    wired: tuple[AsyncClient, MarketDataService],
):
    """A scheduler that silently stopped looks exactly like one that ran and
    found nothing — until a feature window comes up short weeks later."""
    client, _ = wired
    await client.post("/api/v1/market-data/sync?symbols=AAPL")

    resp = await client.get("/api/v1/market-data/sync/status?symbols=AAPL,NEVERFETCHED")
    assert resp.status_code == 200
    body = resp.json()
    by_symbol = {e["symbol"]: e for e in body["symbols"]}
    assert by_symbol["AAPL"]["latest"] is not None
    assert by_symbol["NEVERFETCHED"]["latest"] is None
    # the 2024 fixture bars are years old, and a symbol we hold nothing for is
    # stale by definition — both must be named
    assert set(body["stale"]) == {"AAPL", "NEVERFETCHED"}
