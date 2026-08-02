"""Fundamental features and the point-in-time rule — SHARED definition.

Two things live here for the same reason `features`/`ranking` do: training and
serving must agree bit-for-bit, and a derivation duplicated across a service
boundary is train/serve skew waiting to happen.

1. `fundamental_features` — the numeric encoding of a filing. feature-engine
   merges it into served vectors; ml-pipeline's dataset builder derives it over
   history. One function, one definition.

2. `session_cutoff` / `latest_available_before` — the as-of rule. A filing is
   usable for a decision made during session D only if it was published BEFORE
   that session started. Filings land after the close, so treating a same-day
   10-K as known during the session is intraday look-ahead, and a statement
   with no filing date at all is not usable at any date: undated is not old.
   Getting this wrong does not fail loudly — it produces a model that looks
   good in backtest and has been reading tomorrow's newspaper.
"""

from datetime import UTC, date, datetime, time

from trading_common.schemas import FinancialStatements

# Re-exported: `as_utc` started here, but it is a plain storage-boundary helper
# and market-data needs the same rule. One definition, two importers.
from trading_common.timeutil import as_utc


def _gross_profit(statement: FinancialStatements) -> float | None:
    """Revenue minus cost of revenue, however the filer chose to report it."""
    if statement.gross_profit is not None:
        return statement.gross_profit
    if statement.revenue is not None and statement.cost_of_revenue is not None:
        return statement.revenue - statement.cost_of_revenue
    return None


def fundamental_features(
    statement: FinancialStatements,
    prior: FinancialStatements | None = None,
    price: float | None = None,
) -> dict[str, float]:
    """Scale-free ratios (plus the F-score) from one filing.

    The families named in the prediction plan §5, with the literature each
    comes from. Three of them need more than a single filing, which is why the
    signature takes more than one:

    * `prior` — the previous filing KNOWN AT THE SAME MOMENT, for asset growth.
      Passing the globally-previous filing instead would be look-ahead.
    * `price` — the RAW close of the session being valued, for the two
      valuation ratios. It must be the raw close, not the adjusted one: shares
      outstanding are reported as of the filing, so multiplying them by a
      back-adjusted price computes a market cap that never existed (a later 2:1
      split would halve it).

    Conservative throughout: a missing or degenerate input yields no feature
    rather than a guessed one, so the caller's neutral fill is a visible gap
    instead of a fabricated number. With neither `prior` nor `price` the result
    is exactly the original four features, so existing callers are unaffected.
    """
    out: dict[str, float] = {}
    if statement.piotroski_f_score is not None:
        out["f_score"] = float(statement.piotroski_f_score)
    if statement.net_income is not None and statement.revenue:
        out["fund_net_margin"] = statement.net_income / statement.revenue
    if statement.net_income is not None and statement.total_assets:
        out["fund_roa"] = statement.net_income / statement.total_assets
    if statement.total_liabilities is not None and statement.total_assets:
        out["fund_leverage"] = statement.total_liabilities / statement.total_assets

    # Gross profitability — Novy-Marx 2013. The cleanest profitability measure:
    # further down the income statement every line is more polluted by
    # accounting choices, which is why gross profit beats earnings here.
    gross = _gross_profit(statement)
    if gross is not None and statement.total_assets:
        out["fund_gross_profitability"] = gross / statement.total_assets

    # Accruals — Sloan 1996. Earnings not backed by cash reverse; the ratio is
    # signed on purpose, since the anomaly is that HIGH accruals predict LOW
    # returns.
    if (
        statement.net_income is not None
        and statement.operating_cash_flow is not None
        and statement.total_assets
    ):
        out["fund_accruals"] = (
            statement.net_income - statement.operating_cash_flow
        ) / statement.total_assets

    # Asset growth — Cooper-Gulen-Schill 2008. Companies that expand the balance
    # sheet fastest underperform.
    if prior is not None and statement.total_assets is not None and prior.total_assets:
        out["fund_asset_growth"] = statement.total_assets / prior.total_assets - 1.0

    if price is not None and price > 0 and statement.shares_outstanding:
        market_cap = statement.shares_outstanding * price
        # Book-to-market — Fama-French. Book equity can legitimately be
        # negative (buybacks, accumulated deficits); the ratio stays signed
        # rather than being dropped, because a negative-equity firm is a real
        # and distinct case, not missing data.
        if statement.total_assets is not None and statement.total_liabilities is not None:
            out["fund_book_to_market"] = (
                statement.total_assets - statement.total_liabilities
            ) / market_cap
        # Earnings yield (E/P) — the inverse of the P/E, which is the direction
        # that stays finite when earnings approach zero.
        if statement.net_income is not None:
            out["fund_earnings_yield"] = statement.net_income / market_cap
    return out


FUNDAMENTAL_FEATURE_NAMES: tuple[str, ...] = (
    "f_score",
    "fund_net_margin",
    "fund_roa",
    "fund_leverage",
    "fund_gross_profitability",
    "fund_accruals",
    "fund_asset_growth",
    "fund_book_to_market",
    "fund_earnings_yield",
)


def session_cutoff(day: date) -> datetime:
    """Latest filing instant usable by a decision made during session `day`.

    Midnight UTC of the session: a filing dated that same day does not qualify.
    Losing one day of an annual filing costs nothing measurable; gaining one day
    of hindsight is the classic way a fundamentals backtest lies.
    """
    return datetime.combine(day, time.min, tzinfo=UTC)


def _filing_order(statement: FinancialStatements) -> tuple[datetime, date]:
    # `eligible` has already excluded undated statements, so the assert documents
    # an invariant rather than guarding a real case.
    assert statement.filed_at is not None
    return as_utc(statement.filed_at), statement.period_end


def latest_available_before(
    statements: list[FinancialStatements], cutoff: datetime
) -> FinancialStatements | None:
    """The most recently PUBLISHED statement strictly before `cutoff`.

    Ordered by `filed_at`, never by `period_end`: a later fiscal period filed
    after the cutoff is not knowledge we had. Undated statements are skipped.
    """
    cutoff = as_utc(cutoff)
    eligible = [s for s in statements if s.filed_at is not None and as_utc(s.filed_at) < cutoff]
    if not eligible:
        return None
    return max(eligible, key=_filing_order)


def prior_available_before(
    statements: list[FinancialStatements],
    cutoff: datetime,
    current: FinancialStatements,
) -> FinancialStatements | None:
    """The filing one period back, restricted to what was public at `cutoff`.

    Asset growth compares two balance sheets, and picking the second one by
    period alone would reach for a filing that had not been published yet
    whenever a restatement or a late filer reorders publication against fiscal
    period. Both statements have to clear the same cutoff.
    """
    cutoff = as_utc(cutoff)
    eligible = [
        s
        for s in statements
        if s.filed_at is not None
        and as_utc(s.filed_at) < cutoff
        and s.period_end < current.period_end
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda s: s.period_end)


__all__ = [
    "FUNDAMENTAL_FEATURE_NAMES",
    "as_utc",
    "fundamental_features",
    "latest_available_before",
    "prior_available_before",
    "session_cutoff",
]
