"""Piotroski F-Score from FinancialStatements.

The classic F-Score (Piotroski 2000) sums 9 binary signals. Seven are computable
from the ``FinancialStatements`` contract (revenue, net income, total assets,
total liabilities, operating cash flow). Two require balance-sheet detail the
schema does not carry yet:

- current ratio Δ (needs current assets / current liabilities)
- share issuance (needs shares outstanding)

Those are omitted (documented) until FinancialStatements is extended — so the
score returned here is 0–7. Each signal is conservative: a missing or degenerate
input (None, zero denominator) fails that signal rather than raising.
"""

from dataclasses import dataclass, field

from trading_common.schemas import FinancialStatements


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


@dataclass
class FScoreBreakdown:
    # profitability (current period)
    positive_net_income: bool = False
    positive_operating_cash_flow: bool = False
    quality_of_earnings: bool = False  # OCF > net income (low accruals)
    # profitability trend (needs prior period)
    improving_roa: bool = False
    # leverage / liquidity trend
    decreasing_leverage: bool = False
    # operating efficiency trend
    improving_net_margin: bool = False
    improving_asset_turnover: bool = False

    omitted: tuple[str, ...] = field(
        default=("current_ratio_change", "share_issuance"),
    )

    @property
    def score(self) -> int:
        return sum(
            (
                self.positive_net_income,
                self.positive_operating_cash_flow,
                self.quality_of_earnings,
                self.improving_roa,
                self.decreasing_leverage,
                self.improving_net_margin,
                self.improving_asset_turnover,
            )
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "positive_net_income": self.positive_net_income,
            "positive_operating_cash_flow": self.positive_operating_cash_flow,
            "quality_of_earnings": self.quality_of_earnings,
            "improving_roa": self.improving_roa,
            "decreasing_leverage": self.decreasing_leverage,
            "improving_net_margin": self.improving_net_margin,
            "improving_asset_turnover": self.improving_asset_turnover,
            "omitted": list(self.omitted),
            "score": self.score,
            "max_score": 7,
        }


def compute_f_score(
    current: FinancialStatements,
    prior: FinancialStatements | None = None,
) -> FScoreBreakdown:
    """Compute the (partial) Piotroski F-Score for ``current`` vs ``prior``.

    Without a ``prior`` period, only the three current-period signals can fire.
    """
    b = FScoreBreakdown()

    # --- current-period signals ---
    if current.net_income is not None:
        b.positive_net_income = current.net_income > 0
    if current.operating_cash_flow is not None:
        b.positive_operating_cash_flow = current.operating_cash_flow > 0
    if current.operating_cash_flow is not None and current.net_income is not None:
        b.quality_of_earnings = current.operating_cash_flow > current.net_income

    if prior is None:
        return b

    # --- trend signals (current vs prior) ---
    roa_now = _ratio(current.net_income, current.total_assets)
    roa_prev = _ratio(prior.net_income, prior.total_assets)
    if roa_now is not None and roa_prev is not None:
        b.improving_roa = roa_now > roa_prev

    lev_now = _ratio(current.total_liabilities, current.total_assets)
    lev_prev = _ratio(prior.total_liabilities, prior.total_assets)
    if lev_now is not None and lev_prev is not None:
        b.decreasing_leverage = lev_now < lev_prev

    margin_now = _ratio(current.net_income, current.revenue)
    margin_prev = _ratio(prior.net_income, prior.revenue)
    if margin_now is not None and margin_prev is not None:
        b.improving_net_margin = margin_now > margin_prev

    turnover_now = _ratio(current.revenue, current.total_assets)
    turnover_prev = _ratio(prior.revenue, prior.total_assets)
    if turnover_now is not None and turnover_prev is not None:
        b.improving_asset_turnover = turnover_now > turnover_prev

    return b
