"""The shared fundamental derivation and the point-in-time rule (P2-3).

The as-of rule is the load-bearing part: joining today's F-score onto a 2022
session teaches the model facts published two years later, and that failure is
invisible — it makes the backtest look BETTER.
"""

from datetime import UTC, date, datetime

import pytest

from trading_common.fundamentals import (
    fundamental_features,
    latest_available_before,
    prior_available_before,
    session_cutoff,
)
from trading_common.schemas import FinancialStatements


def statement(
    period_end: str,
    filed_at: str | None,
    *,
    symbol: str = "AAPL",
    f_score: int | None = 6,
    revenue: float | None = 1000.0,
    net_income: float | None = 100.0,
    assets: float | None = 2000.0,
    liabilities: float | None = 800.0,
) -> FinancialStatements:
    return FinancialStatements(
        symbol=symbol,
        period_end=date.fromisoformat(period_end),
        fiscal_period="FY",
        filed_at=datetime.fromisoformat(filed_at).replace(tzinfo=UTC) if filed_at else None,
        piotroski_f_score=f_score,
        revenue=revenue,
        net_income=net_income,
        total_assets=assets,
        total_liabilities=liabilities,
    )


# --- the derivation ---------------------------------------------------------


def test_features_are_scale_free_ratios_plus_the_score():
    feats = fundamental_features(statement("2024-12-31", "2025-02-10"))
    assert feats["f_score"] == 6.0
    assert feats["fund_net_margin"] == pytest.approx(0.1)
    assert feats["fund_roa"] == pytest.approx(0.05)
    assert feats["fund_leverage"] == pytest.approx(0.4)


def test_missing_or_degenerate_inputs_yield_no_feature():
    """No fabricated zeros: the caller's neutral fill must stay a visible gap."""
    empty = fundamental_features(
        statement("2024-12-31", "2025-02-10", f_score=None, net_income=None, assets=None)
    )
    # every ratio here needs total_assets or net_income — none survive
    assert empty == {}

    zero_revenue = fundamental_features(statement("2024-12-31", "2025-02-10", revenue=0.0))
    assert "fund_net_margin" not in zero_revenue  # division by zero, not infinity


# --- the point-in-time rule -------------------------------------------------


PANEL = [
    statement("2022-12-31", "2023-02-03"),
    statement("2023-12-31", "2024-02-02"),
    statement("2024-12-31", "2025-02-07"),
]


def test_as_of_returns_what_was_published_not_what_exists():
    """The whole defect P2-3 closes, in one assertion."""
    picked = latest_available_before(PANEL, session_cutoff(date(2024, 6, 14)))
    assert picked is not None
    assert picked.period_end == date(2023, 12, 31)  # NOT the 2024 filing


def test_a_filing_is_not_usable_on_its_own_filing_day():
    """Filings land after the close, so counting one as known during that
    session is intraday look-ahead. One day of an annual filing costs nothing;
    one day of hindsight is how a fundamentals backtest lies."""
    filed_day = date(2024, 2, 2)
    assert latest_available_before(PANEL, session_cutoff(filed_day)) is not None
    assert latest_available_before(PANEL, session_cutoff(filed_day)).period_end == date(
        2022, 12, 31
    )
    # the next session does see it
    next_day = latest_available_before(PANEL, session_cutoff(date(2024, 2, 5)))
    assert next_day is not None and next_day.period_end == date(2023, 12, 31)


def test_nothing_published_yet_returns_none():
    assert latest_available_before(PANEL, session_cutoff(date(2020, 1, 2))) is None
    assert latest_available_before([], session_cutoff(date(2024, 1, 2))) is None


def test_undated_statements_are_invisible_not_ancient():
    """A statement we cannot place in time is unusable at EVERY date. Treating
    it as very old would quietly reintroduce the look-ahead — it would win the
    join for every session before the first dated filing."""
    undated = statement("2021-12-31", None)
    picked = latest_available_before([undated], session_cutoff(date(2024, 6, 14)))
    assert picked is None
    mixed = latest_available_before([undated, *PANEL], session_cutoff(date(2023, 6, 14)))
    assert mixed is not None and mixed.period_end == date(2022, 12, 31)


def test_ordering_is_by_filing_date_not_period_end():
    """A restated 2020 report filed in 2025 is 2025 knowledge, not 2020's; and a
    later fiscal period filed after the cutoff must not win on period_end."""
    late_restatement = statement("2020-12-31", "2025-06-01", f_score=1)
    picked = latest_available_before([*PANEL, late_restatement], session_cutoff(date(2025, 7, 1)))
    assert picked is not None
    assert picked.period_end == date(2020, 12, 31)  # newest FILING wins
    assert picked.piotroski_f_score == 1


def test_session_cutoff_is_midnight_utc():
    cutoff = session_cutoff(date(2024, 3, 14))
    assert cutoff == datetime(2024, 3, 14, 0, 0, tzinfo=UTC)


def test_naive_and_aware_timestamps_compare_without_crashing():
    """`filed_at` arrives aware from Postgres, naive from sqlite, and either way
    from a hand-posted statement. Comparing the two kinds raises TypeError, so
    without normalization the point-in-time join would crash on one backend and
    quietly work on the other."""
    naive = FinancialStatements(
        symbol="AAPL",
        period_end=date(2023, 12, 31),
        fiscal_period="FY",
        filed_at=datetime(2024, 2, 2),  # no tzinfo
        piotroski_f_score=7,
    )
    picked = latest_available_before([naive], session_cutoff(date(2024, 6, 14)))
    assert picked is not None and picked.piotroski_f_score == 7
    # ...and the cutoff itself may be naive too
    assert latest_available_before([naive], datetime(2024, 6, 14)) is not None
    assert latest_available_before([naive], datetime(2024, 1, 1)) is None


# --- the factor families from the prediction plan §5 ------------------------


def _filing(
    period_end: date,
    filed: datetime | None = None,
    **fields: float,
) -> FinancialStatements:
    return FinancialStatements(
        symbol="AAPL", period_end=period_end, fiscal_period="FY", filed_at=filed, **fields
    )


def test_gross_profitability_prefers_the_reported_figure_and_derives_it_otherwise():
    """Novy-Marx 2013. Filers report GrossProfit or CostOfRevenue, not both by
    convention — a factor available on half the universe is not a factor."""
    direct = _filing(date(2024, 12, 31), revenue=1000, gross_profit=400, total_assets=2000)
    assert fundamental_features(direct)["fund_gross_profitability"] == pytest.approx(0.2)

    derived = _filing(date(2024, 12, 31), revenue=1000, cost_of_revenue=650, total_assets=2000)
    assert fundamental_features(derived)["fund_gross_profitability"] == pytest.approx(0.175)

    neither = _filing(date(2024, 12, 31), revenue=1000, total_assets=2000)
    assert "fund_gross_profitability" not in fundamental_features(neither)


def test_accruals_keep_their_sign():
    """Sloan 1996: HIGH accruals predict LOW returns, so the ratio must stay
    signed. Taking an absolute value would merge the two ends of the anomaly."""
    earnings_without_cash = _filing(
        date(2024, 12, 31), net_income=200, operating_cash_flow=50, total_assets=1000
    )
    cash_rich = _filing(
        date(2024, 12, 31), net_income=200, operating_cash_flow=350, total_assets=1000
    )
    assert fundamental_features(earnings_without_cash)["fund_accruals"] == pytest.approx(0.15)
    assert fundamental_features(cash_rich)["fund_accruals"] == pytest.approx(-0.15)


def test_asset_growth_needs_a_prior_and_says_nothing_without_one():
    current = _filing(date(2024, 12, 31), total_assets=1200)
    prior = _filing(date(2023, 12, 31), total_assets=1000)
    assert fundamental_features(current, prior=prior)["fund_asset_growth"] == pytest.approx(0.2)
    assert "fund_asset_growth" not in fundamental_features(current)


def test_valuation_ratios_use_the_market_cap_at_that_price():
    current = _filing(
        date(2024, 12, 31),
        total_assets=1000,
        total_liabilities=600,
        net_income=80,
        shares_outstanding=100,
    )
    out = fundamental_features(current, price=8.0)  # market cap 800
    assert out["fund_book_to_market"] == pytest.approx(0.5)  # equity 400 / 800
    assert out["fund_earnings_yield"] == pytest.approx(0.1)  # 80 / 800
    # ...and no price means no guess
    assert "fund_book_to_market" not in fundamental_features(current)


def test_negative_book_equity_is_reported_not_dropped():
    """Buybacks and accumulated deficits produce negative book equity. That is a
    real and distinct case, not missing data — dropping it would quietly remove
    a whole class of company from the factor."""
    levered = _filing(
        date(2024, 12, 31), total_assets=1000, total_liabilities=1400, shares_outstanding=100
    )
    assert fundamental_features(levered, price=10.0)["fund_book_to_market"] < 0


def test_the_original_four_features_are_unchanged_without_prior_or_price():
    """Serving passes neither, and must keep computing exactly what it did."""
    statement = _filing(
        date(2024, 12, 31),
        revenue=1000,
        net_income=100,
        total_assets=2000,
        total_liabilities=800,
    )
    assert set(fundamental_features(statement)) == {
        "fund_net_margin",
        "fund_roa",
        "fund_leverage",
    }


def test_the_prior_must_clear_the_same_cutoff_as_the_current_filing():
    """The trap: picking the previous filing by fiscal period alone reaches for
    a statement that had not been published yet whenever a restatement or a late
    filer reorders publication against period. Asset growth would then be
    computed from a balance sheet nobody had seen."""
    cutoff = datetime(2025, 4, 1, tzinfo=UTC)
    fy2024 = _filing(date(2024, 12, 31), filed=datetime(2025, 2, 1, tzinfo=UTC), total_assets=1200)
    fy2023 = _filing(date(2023, 12, 31), filed=datetime(2024, 2, 1, tzinfo=UTC), total_assets=1000)
    # filed AFTER the cutoff even though its period sits in between
    restated = _filing(date(2024, 6, 30), filed=datetime(2025, 6, 1, tzinfo=UTC), total_assets=9999)

    prior = prior_available_before([fy2024, fy2023, restated], cutoff, fy2024)
    assert prior is not None
    assert prior.period_end == date(2023, 12, 31), "an unpublished filing was used as the prior"


def test_an_undated_filing_is_never_a_prior():
    """Undated is not old — the same rule the as-of read already enforces."""
    cutoff = datetime(2025, 4, 1, tzinfo=UTC)
    current = _filing(date(2024, 12, 31), filed=datetime(2025, 2, 1, tzinfo=UTC), total_assets=1200)
    undated = _filing(date(2023, 12, 31), total_assets=1000)
    assert prior_available_before([current, undated], cutoff, current) is None
