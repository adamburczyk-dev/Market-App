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
