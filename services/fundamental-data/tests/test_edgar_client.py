"""Tests for the SEC EDGAR client (via httpx MockTransport)."""

from datetime import date

import httpx
import pytest

from src.core.edgar_client import EdgarClient

TICKERS = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}

# concept tag → [(end, val, form, fp), ...]  (two annual periods)
CONCEPTS = {
    "Revenues": [("2023-09-30", 383285, "10-K", "FY"), ("2024-09-28", 391035, "10-K", "FY")],
    "NetIncomeLoss": [("2023-09-30", 96995, "10-K", "FY"), ("2024-09-28", 93736, "10-K", "FY")],
    "Assets": [("2023-09-30", 352583, "10-K", "FY"), ("2024-09-28", 364980, "10-K", "FY")],
    "Liabilities": [("2023-09-30", 290437, "10-K", "FY"), ("2024-09-28", 308030, "10-K", "FY")],
    "NetCashProvidedByUsedInOperatingActivities": [
        ("2023-09-30", 110543, "10-K", "FY"),
        ("2024-09-28", 118254, "10-K", "FY"),
    ],
    "EarningsPerShareBasic": [
        ("2023-09-30", 6.16, "10-K", "FY"),
        ("2024-09-28", 6.11, "10-K", "FY"),
    ],
}


def sec_handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path.endswith("company_tickers.json"):
        return httpx.Response(200, json=TICKERS)
    if "/companyconcept/" in path:
        tag = path.rsplit("/", 1)[-1].removesuffix(".json")
        unit = "USD/shares" if tag == "EarningsPerShareBasic" else "USD"
        obs = [{"end": e, "val": v, "form": f, "fp": fp} for (e, v, f, fp) in CONCEPTS.get(tag, [])]
        return httpx.Response(200, json={"units": {unit: obs}})
    return httpx.Response(404)


def client_with(handler, user_agent="test-agent contact@example.com"):  # type: ignore[no-untyped-def]
    ec = EdgarClient(user_agent)
    ec._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return ec


@pytest.mark.asyncio
async def test_disabled_without_user_agent():
    ec = EdgarClient(None)
    assert ec.enabled is False
    assert await ec.latest_statements("AAPL") == []
    await ec.aclose()


@pytest.mark.asyncio
async def test_ticker_to_cik_zero_padded():
    ec = client_with(sec_handler)
    assert await ec.ticker_to_cik("AAPL") == "0000320193"
    assert await ec.ticker_to_cik("aapl") == "0000320193"  # case-insensitive
    assert await ec.ticker_to_cik("ZZZZ") is None
    await ec.aclose()


@pytest.mark.asyncio
async def test_latest_statements_assembles_two_periods():
    ec = client_with(sec_handler)
    statements = await ec.latest_statements("AAPL", count=2)
    await ec.aclose()
    assert len(statements) == 2
    # most-recent first
    assert statements[0].period_end == date(2024, 9, 28)
    assert statements[1].period_end == date(2023, 9, 30)
    latest = statements[0]
    assert latest.revenue == 391035
    assert latest.net_income == 93736
    assert latest.total_assets == 364980
    assert latest.operating_cash_flow == 118254
    assert latest.eps == 6.11
    assert latest.source == "sec-edgar"
    assert latest.fiscal_period == "FY"


@pytest.mark.asyncio
async def test_unknown_ticker_returns_empty():
    ec = client_with(sec_handler)
    assert await ec.latest_statements("ZZZZ") == []
    await ec.aclose()


@pytest.mark.asyncio
async def test_non_annual_filings_ignored():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("company_tickers.json"):
            return httpx.Response(200, json=TICKERS)
        # only a quarterly filing → must be skipped
        return httpx.Response(
            200,
            json={"units": {"USD": [{"end": "2024-06-30", "val": 1, "form": "10-Q", "fp": "Q3"}]}},
        )

    ec = client_with(handler)
    assert await ec.latest_statements("AAPL") == []
    await ec.aclose()


@pytest.mark.asyncio
async def test_http_error_yields_no_statements():
    ec = client_with(lambda r: httpx.Response(500))
    assert await ec.latest_statements("AAPL") == []
    await ec.aclose()
