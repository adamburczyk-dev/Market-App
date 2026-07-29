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


def fundamental_features(statement: FinancialStatements) -> dict[str, float]:
    """Scale-free ratios (plus the F-score) from one filing.

    Conservative throughout: a missing or degenerate input yields no feature
    rather than a guessed one, so the caller's neutral fill is a visible gap
    instead of a fabricated number.
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
    return out


FUNDAMENTAL_FEATURE_NAMES: tuple[str, ...] = (
    "f_score",
    "fund_net_margin",
    "fund_roa",
    "fund_leverage",
)


def session_cutoff(day: date) -> datetime:
    """Latest filing instant usable by a decision made during session `day`.

    Midnight UTC of the session: a filing dated that same day does not qualify.
    Losing one day of an annual filing costs nothing measurable; gaining one day
    of hindsight is the classic way a fundamentals backtest lies.
    """
    return datetime.combine(day, time.min, tzinfo=UTC)


def as_utc(moment: datetime) -> datetime:
    """Treat a naive timestamp as UTC.

    Not cosmetic: `filed_at` reaches this rule from a Postgres TIMESTAMPTZ
    (aware), from sqlite (naive — the driver drops the zone), and from a
    hand-posted statement (either). Comparing a naive to an aware datetime
    raises TypeError, so without this the point-in-time join would crash on one
    backend and work on another. UTC is the right assumption: every producer in
    this system writes UTC.
    """
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


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


__all__ = [
    "FUNDAMENTAL_FEATURE_NAMES",
    "as_utc",
    "fundamental_features",
    "latest_available_before",
    "session_cutoff",
]
