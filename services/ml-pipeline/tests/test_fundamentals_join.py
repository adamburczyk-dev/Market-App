"""The point-in-time fundamentals join in training (P2-3).

The defect these guard against does not raise: joining the newest filing onto
every session makes the model look BETTER, because it hands it facts published
months later. So the tests assert the value that landed in the row, not that
the code ran.
"""

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest
from trading_common.schemas import FinancialStatements, Interval, OHLCVBar

from src.core.dataset import DatasetParams, build_dataset
from src.core.fundamentals_client import HttpFundamentalsClient

START = datetime(2021, 1, 4, tzinfo=UTC)
N_BARS = 400
PARAMS = DatasetParams(min_universe=20)


def bars(symbol: str, seed: int) -> list[OHLCVBar]:
    rng = np.random.default_rng(seed)
    closes = 100.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.012, N_BARS))
    return [
        OHLCVBar(
            symbol=symbol,
            timestamp=START + timedelta(days=i),
            interval=Interval.D1,
            open=float(c),
            high=float(c) * 1.01,
            low=float(c) * 0.99,
            close=float(c),
            adj_close=float(c),
            volume=1_000_000.0 + i,
            source="test",
        )
        for i, c in enumerate(closes)
    ]


def universe(n: int = 24) -> dict[str, list[OHLCVBar]]:
    return {f"S{k:02d}": bars(f"S{k:02d}", k + 1) for k in range(n)}


def statement(symbol: str, period_end: str, filed_at: str | None, f_score: int):
    return FinancialStatements(
        symbol=symbol,
        period_end=date.fromisoformat(period_end),
        fiscal_period="FY",
        filed_at=datetime.fromisoformat(filed_at).replace(tzinfo=UTC) if filed_at else None,
        revenue=1000.0,
        net_income=100.0,
        total_assets=2000.0,
        total_liabilities=800.0,
        piotroski_f_score=f_score,
    )


def panel(symbols: list[str]) -> dict[str, list[FinancialStatements]]:
    """Two filings per symbol: an early weak one and a later strong one."""
    return {
        s: [
            statement(s, "2020-12-31", "2021-02-15", 2),
            statement(s, "2021-12-31", "2022-02-15", 8),
        ]
        for s in symbols
    }


def column(ds, name: str) -> np.ndarray:
    return ds.x[:, ds.feature_names.index(name)]


def test_fundamentals_enter_the_feature_contract_when_requested():
    u = universe()
    plain = build_dataset(u, PARAMS)
    joined = build_dataset(u, PARAMS, fundamentals_by_symbol=panel(sorted(u)))

    assert "f_score" not in plain.feature_names
    for name in ("f_score", "fund_net_margin", "fund_roa", "fund_leverage"):
        assert name in joined.feature_names
    assert plain.fundamental_coverage == 0.0
    assert joined.fundamental_coverage > 0.9  # both filings precede most sessions


def test_a_session_only_sees_filings_published_before_it():
    """The whole point. Every symbol has the same two filings, so the F-score is
    constant across the cross-section on any given session — which means its
    RANK is uninformative, but its presence is exactly datable. Sessions before
    the first filing must carry no fundamental value at all."""
    u = universe()
    # one symbol's later filing lands mid-history; the rest keep the early one
    marked = "S00"
    p = panel(sorted(u))
    # the second filing lands mid-history so rows exist on both sides of it
    p[marked] = [
        statement(marked, "2020-12-31", "2021-02-15", 1),
        statement(marked, "2021-06-30", "2021-11-15", 9),
    ]
    for other in list(p):
        if other != marked:
            p[other] = [statement(other, "2020-12-31", "2021-02-15", 5)]

    ds = build_dataset(u, PARAMS, fundamentals_by_symbol=p)
    scores = column(ds, "f_score")
    marked_rows = [i for i, s in enumerate(ds.symbols) if s == marked]
    before = [i for i in marked_rows if ds.dates[i] < datetime(2021, 11, 15, tzinfo=UTC)]
    after = [i for i in marked_rows if ds.dates[i] > datetime(2021, 11, 16, tzinfo=UTC)]

    assert before and after, "the fixture must straddle the second filing"
    # F-score 1 is the lowest in the universe (others are 5) → rank 0.0;
    # after the second filing it is the highest (9) → rank 1.0.
    assert scores[before].max() < 0.5
    assert scores[after].min() > 0.5


def test_an_undated_filing_is_never_joined():
    """Undated is not old. If it could win the join it would be 'known' for the
    whole history — the worst possible look-ahead."""
    u = universe()
    p = {s: [statement(s, "2019-12-31", None, 9)] for s in sorted(u)}
    ds = build_dataset(u, PARAMS, fundamentals_by_symbol=p)
    assert ds.fundamental_coverage == 0.0
    assert "f_score" not in ds.feature_names  # nothing was ever merged


def test_symbols_without_a_filing_are_neutral_filled_and_counted():
    """Half the universe has fundamentals. The missing half must be neutral 0.5
    — a visible gap — and the coverage number must say so, because a column
    that is mostly placeholder looks identical to a weak feature."""
    u = universe()
    covered = sorted(u)[:12]
    ds = build_dataset(u, PARAMS, fundamentals_by_symbol=panel(covered))
    assert 0.4 < ds.fundamental_coverage < 0.6
    missing = [i for i, s in enumerate(ds.symbols) if s not in covered]
    assert all(column(ds, "f_score")[i] == 0.5 for i in missing)


def test_coverage_reaches_the_data_contract_report():
    from src.core.data_contract import build_report

    u = universe()
    ds = build_dataset(u, PARAMS, fundamentals_by_symbol=panel(sorted(u)))
    report = build_report(ds)
    assert report["fundamental_coverage"] == ds.fundamental_coverage > 0.9


# --- the client ------------------------------------------------------------


@pytest.mark.asyncio
async def test_client_groups_the_panel_by_symbol(monkeypatch):
    import httpx

    body = {
        "symbols": ["AAPL", "MSFT"],
        "rows": 3,
        "rows_without_filed_at": 1,
        "statements": [
            statement("AAPL", "2022-12-31", "2023-02-03", 5).model_dump(mode="json"),
            statement("AAPL", "2023-12-31", "2024-02-02", 7).model_dump(mode="json"),
            statement("MSFT", "2023-06-30", None, 4).model_dump(mode="json"),
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["symbols"] == "AAPL,MSFT"
        return httpx.Response(200, json=body)

    client = HttpFundamentalsClient("http://fundamental-data:8000")
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    panel_by_symbol = await client.panel(["AAPL", "MSFT"])
    await client.aclose()

    assert set(panel_by_symbol) == {"AAPL", "MSFT"}
    assert len(panel_by_symbol["AAPL"]) == 2
    assert panel_by_symbol["MSFT"][0].filed_at is None  # stored, but unusable


@pytest.mark.asyncio
async def test_client_degrades_instead_of_failing_the_run():
    """No panel is a legitimate state (no database, no EDGAR access). The caller
    reports the resulting coverage; an exception would turn a data gap into a
    crashed training run."""
    import httpx

    client = HttpFundamentalsClient("http://fundamental-data:8000")
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(503)))
    assert await client.panel(["AAPL"]) == {}
    await client.aclose()


# --- the factor families from the prediction plan §5 ------------------------


def valued_panel(symbols: list[str]) -> dict[str, list[FinancialStatements]]:
    """Two filings per symbol, carrying everything the new factors need.

    Both filing dates sit INSIDE the bar window (which starts 2021-01-04 and
    runs 400 days): asset growth compares two balance sheets, so a fixture
    whose second filing is published after the last session can only ever show
    one of them — and the test would pass or fail for the wrong reason.
    """
    out: dict[str, list[FinancialStatements]] = {}
    for index, s in enumerate(symbols):
        base = 1000.0 + 50.0 * index  # a real cross-section, not one constant
        # growth must DIFFER across names too: an identical ratio everywhere
        # ranks to 0.5 for all of them, which looks exactly like a neutral fill
        growth = 1.05 + 0.02 * index
        out[s] = [
            statement(s, "2020-06-30", "2021-02-15", 2).model_copy(
                update={
                    "total_assets": base,
                    "gross_profit": 300.0,
                    "operating_cash_flow": 60.0,
                    "shares_outstanding": 100.0 + index,
                }
            ),
            statement(s, "2021-06-30", "2021-11-01", 8).model_copy(
                update={
                    "total_assets": base * growth,
                    "gross_profit": 400.0,
                    "operating_cash_flow": 90.0,
                    "shares_outstanding": 100.0 + index,
                }
            ),
        ]
    return out


def test_every_named_factor_family_reaches_the_feature_contract():
    """The §5 table: the families were documented long before they were
    computed, and the run that mattered had none of them."""
    u = universe()
    ds = build_dataset(u, PARAMS, fundamentals_by_symbol=valued_panel(sorted(u)))
    for name in (
        "fund_gross_profitability",
        "fund_accruals",
        "fund_asset_growth",
        "fund_book_to_market",
        "fund_earnings_yield",
    ):
        assert name in ds.feature_names, f"{name} never reached the dataset"
        # ranked into [0, 1] and not a single constant value
        values = column(ds, name)
        assert values.min() >= 0.0 and values.max() <= 1.0
        assert values.std() > 0.0, f"{name} is constant — it carries nothing"


def test_asset_growth_is_absent_while_only_one_filing_is_public():
    """Before the second filing there is no prior to compare against, and the
    honest answer is a neutral fill rather than a growth rate invented from the
    one balance sheet that exists."""
    u = universe()
    single = {s: [v[0]] for s, v in valued_panel(sorted(u)).items()}
    ds = build_dataset(u, PARAMS, fundamentals_by_symbol=single)
    assert "fund_asset_growth" not in ds.feature_names


def test_the_valuation_ratio_moves_with_price_not_just_with_the_filing():
    """B/M and E/P are the only fundamental features that change between
    filings. If they were computed from the statement alone they would be step
    functions, and the rank would be a rank of filing dates."""
    u = universe()
    ds = build_dataset(u, PARAMS, fundamentals_by_symbol=valued_panel(sorted(u)))
    first_symbol = ds.symbols[0]
    rows = [i for i, s in enumerate(ds.symbols) if s == first_symbol]
    series = column(ds, "fund_book_to_market")[rows]
    assert len(set(series.round(6))) > 1, "book-to-market never moved for a symbol"
