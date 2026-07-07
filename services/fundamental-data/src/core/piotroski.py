"""Piotroski F-Score from FinancialStatements.

The classic F-Score (Piotroski 2000) sums 9 binary signals; all are computable
now that ``FinancialStatements`` carries balance-sheet detail (current assets /
current liabilities for the liquidity trend, shares outstanding for dilution).
Each signal is conservative: a missing or degenerate input (None, zero
denominator) fails that signal rather than raising — so a statement without
the balance-sheet detail simply cannot score above 7.
"""

from dataclasses import dataclass

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
    improving_current_ratio: bool = False
    # dilution
    no_dilution: bool = False  # shares outstanding did not grow
    # operating efficiency trend
    improving_net_margin: bool = False
    improving_asset_turnover: bool = False

    @property
    def score(self) -> int:
        return sum(
            (
                self.positive_net_income,
                self.positive_operating_cash_flow,
                self.quality_of_earnings,
                self.improving_roa,
                self.decreasing_leverage,
                self.improving_current_ratio,
                self.no_dilution,
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
            "improving_current_ratio": self.improving_current_ratio,
            "no_dilution": self.no_dilution,
            "improving_net_margin": self.improving_net_margin,
            "improving_asset_turnover": self.improving_asset_turnover,
            "score": self.score,
            "max_score": 9,
        }


def compute_f_score(
    current: FinancialStatements,
    prior: FinancialStatements | None = None,
) -> FScoreBreakdown:
    """Compute the Piotroski F-Score (0-9) for ``current`` vs ``prior``.

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

    cr_now = _ratio(current.current_assets, current.current_liabilities)
    cr_prev = _ratio(prior.current_assets, prior.current_liabilities)
    if cr_now is not None and cr_prev is not None:
        b.improving_current_ratio = cr_now > cr_prev

    if current.shares_outstanding is not None and prior.shares_outstanding is not None:
        b.no_dilution = current.shares_outstanding <= prior.shares_outstanding

    margin_now = _ratio(current.net_income, current.revenue)
    margin_prev = _ratio(prior.net_income, prior.revenue)
    if margin_now is not None and margin_prev is not None:
        b.improving_net_margin = margin_now > margin_prev

    turnover_now = _ratio(current.revenue, current.total_assets)
    turnover_prev = _ratio(prior.revenue, prior.total_assets)
    if turnover_now is not None and turnover_prev is not None:
        b.improving_asset_turnover = turnover_now > turnover_prev

    return b
