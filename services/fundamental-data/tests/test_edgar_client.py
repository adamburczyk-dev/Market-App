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
    "AssetsCurrent": [("2023-09-30", 143566, "10-K", "FY"), ("2024-09-28", 152987, "10-K", "FY")],
    "LiabilitiesCurrent": [
        ("2023-09-30", 145308, "10-K", "FY"),
        ("2024-09-28", 176392, "10-K", "FY"),
    ],
    "CommonStockSharesOutstanding": [
        ("2023-09-30", 15550, "10-K", "FY"),
        ("2024-09-28", 15116, "10-K", "FY"),
    ],
    "NetCashProvidedByUsedInOperatingActivities": [
        ("2023-09-30", 110543, "10-K", "FY"),
        ("2024-09-28", 118254, "10-K", "FY"),
    ],
    "EarningsPerShareBasic": [
        ("2023-09-30", 6.16, "10-K", "FY"),
        ("2024-09-28", 6.11, "10-K", "FY"),
    ],
}

TAG_UNITS = {
    "EarningsPerShareBasic": "USD/shares",
    "CommonStockSharesOutstanding": "shares",
    "WeightedAverageNumberOfSharesOutstandingBasic": "shares",
    "WeightedAverageNumberOfDilutedSharesOutstanding": "shares",
}


def sec_handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path.endswith("company_tickers.json"):
        return httpx.Response(200, json=TICKERS)
    if "/companyconcept/" in path:
        tag = path.rsplit("/", 1)[-1].removesuffix(".json")
        unit = TAG_UNITS.get(tag, "USD")
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
    assert latest.current_assets == 152987
    assert latest.current_liabilities == 176392
    assert latest.shares_outstanding == 15116  # buyback vs 15550 prior
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


# --- P3: revenue tag fallbacks (ASC 606 filers don't report ``Revenues``) ---


def handler_with_concepts(concepts: dict):  # type: ignore[no-untyped-def]
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("company_tickers.json"):
            return httpx.Response(200, json=TICKERS)
        if "/companyconcept/" in path:
            tag = path.rsplit("/", 1)[-1].removesuffix(".json")
            if tag not in concepts:
                return httpx.Response(404)  # filer doesn't use this concept
            unit = "USD/shares" if tag == "EarningsPerShareBasic" else "USD"
            obs = [{"end": e, "val": v, "form": f, "fp": fp} for (e, v, f, fp) in concepts[tag]]
            return httpx.Response(200, json={"units": {unit: obs}})
        return httpx.Response(404)

    return handler


@pytest.mark.asyncio
async def test_asc606_filer_falls_back_to_contract_revenue_tag():
    concepts = {
        "RevenueFromContractWithCustomerExcludingAssessedTax": [
            ("2024-09-28", 391035, "10-K", "FY")
        ],
        "NetIncomeLoss": [("2024-09-28", 93736, "10-K", "FY")],
    }
    ec = client_with(handler_with_concepts(concepts))
    statements = await ec.latest_statements("AAPL", count=1)
    await ec.aclose()
    assert statements[0].revenue == 391035


@pytest.mark.asyncio
async def test_primary_revenue_tag_wins_over_fallback():
    concepts = {
        "Revenues": [("2024-09-28", 400000, "10-K", "FY")],
        "RevenueFromContractWithCustomerExcludingAssessedTax": [
            ("2024-09-28", 391035, "10-K", "FY")
        ],
    }
    ec = client_with(handler_with_concepts(concepts))
    statements = await ec.latest_statements("AAPL", count=1)
    await ec.aclose()
    assert statements[0].revenue == 400000  # earlier candidate takes priority


@pytest.mark.asyncio
async def test_tag_switch_across_periods_unions_both():
    # pre-ASC-606 year under Revenues, post-switch year under the contract tag
    concepts = {
        "Revenues": [("2023-09-30", 383285, "10-K", "FY")],
        "RevenueFromContractWithCustomerExcludingAssessedTax": [
            ("2024-09-28", 391035, "10-K", "FY")
        ],
    }
    ec = client_with(handler_with_concepts(concepts))
    statements = await ec.latest_statements("AAPL", count=2)
    await ec.aclose()
    assert statements[0].revenue == 391035  # 2024 from the fallback tag
    assert statements[1].revenue == 383285  # 2023 from the classic tag


# --- P2-3: filed_at, the date that makes the panel point-in-time -----------


FILED = {
    # same period reported three times: the original 10-K, an amendment, and
    # again as the comparative column of the next year's filing
    "Revenues": [
        ("2023-09-30", 383285, "10-K", "FY", "2023-11-03"),
        ("2023-09-30", 383285, "10-K/A", "FY", "2024-01-15"),
        ("2023-09-30", 383285, "10-K", "FY", "2024-11-01"),
    ],
    "NetIncomeLoss": [("2023-09-30", 96995, "10-K", "FY", "2023-11-03")],
    "Assets": [("2023-09-30", 352583, "10-K", "FY", "2023-11-20")],
}


def filed_handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path.endswith("company_tickers.json"):
        return httpx.Response(200, json=TICKERS)
    if "/companyconcept/" in path:
        tag = path.rsplit("/", 1)[-1].removesuffix(".json")
        obs = [
            {"end": e, "val": v, "form": f, "fp": fp, "filed": filed}
            for (e, v, f, fp, filed) in FILED.get(tag, [])
        ]
        return httpx.Response(200, json={"units": {"USD": obs}})
    return httpx.Response(404)


@pytest.mark.asyncio
async def test_filed_at_is_the_last_field_to_become_public():
    """A statement is knowable once every field it carries has been published.

    Revenue first appeared 2023-11-03 and Assets 2023-11-20, so the statement
    was complete on the 20th. Dating it by the earliest field would claim
    knowledge of a number that was not out yet — being late costs a little
    information, being early fabricates it.
    """
    ec = client_with(filed_handler)
    statements = await ec.latest_statements("AAPL", count=1)
    await ec.aclose()
    assert statements[0].filed_at is not None
    assert statements[0].filed_at.date() == date(2023, 11, 20)


@pytest.mark.asyncio
async def test_the_earliest_filing_of_a_value_wins_not_the_latest():
    """The same period is re-reported by amendments and by next year's
    comparatives. What matters is when the market FIRST had it."""
    ec = client_with(filed_handler)
    statements = await ec.latest_statements("AAPL", count=1)
    await ec.aclose()
    # 2024-11-01 and 2024-01-15 also report revenue for this period; neither
    # may push the statement's date forward past the Assets filing.
    assert statements[0].filed_at.date() == date(2023, 11, 20)


@pytest.mark.asyncio
async def test_a_field_without_a_filing_date_leaves_the_statement_undated():
    """Fail closed: one undated field means the statement cannot be placed in
    time, and the as-of read then skips it entirely rather than guessing."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("company_tickers.json"):
            return httpx.Response(200, json=TICKERS)
        tag = request.url.path.rsplit("/", 1)[-1].removesuffix(".json")
        rows = FILED.get(tag, [])
        obs = [
            {"end": e, "val": v, "form": f, "fp": fp}  # no "filed"
            if tag == "Assets"
            else {"end": e, "val": v, "form": f, "fp": fp, "filed": filed}
            for (e, v, f, fp, filed) in rows
        ]
        return httpx.Response(200, json={"units": {"USD": obs}})

    ec = client_with(handler)
    statements = await ec.latest_statements("AAPL", count=1)
    await ec.aclose()
    assert statements[0].total_assets == 352583  # the value is still there
    assert statements[0].filed_at is None  # ...but it is not dateable


@pytest.mark.asyncio
async def test_statements_without_any_filing_dates_stay_undated():
    # the original fixture has no "filed" fields at all
    ec = client_with(sec_handler)
    statements = await ec.latest_statements("AAPL", count=1)
    await ec.aclose()
    assert statements[0].filed_at is None
