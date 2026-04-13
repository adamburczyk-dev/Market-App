"""Drawdown-adaptive position sizing — anti-martingale approach."""


class DrawdownAdaptiveSizer:
    """
    Scales position risk inversely with current drawdown.
    At peak equity: full risk budget. At max drawdown: zero new positions.
    Linear interpolation between dd_scaling_start and dd_scaling_end.

    Research: Vince (1990) "Portfolio Management Formulas",
    Tharp (2008) "Position Sizing" — anti-martingale approach.
    """

    def __init__(
        self,
        base_risk_per_trade: float = 0.02,
        dd_scaling_start: float = 0.05,
        dd_scaling_end: float = 0.15,
    ) -> None:
        self.base_risk = base_risk_per_trade
        self.dd_start = dd_scaling_start
        self.dd_end = dd_scaling_end

    def compute_risk_budget(self, current_drawdown_pct: float) -> float:
        """
        Returns fraction of portfolio to risk on next trade.
        Range: [0.0, base_risk_per_trade].
        """
        dd = abs(current_drawdown_pct)
        if dd <= self.dd_start:
            return self.base_risk
        if dd >= self.dd_end:
            return 0.0
        scale = 1.0 - (dd - self.dd_start) / (self.dd_end - self.dd_start)
        return self.base_risk * scale

    def position_size(
        self,
        portfolio_value: float,
        entry_price: float,
        stop_loss_price: float,
        current_drawdown_pct: float,
    ) -> int:
        """Returns number of shares to buy. Capped at 5% of portfolio."""
        risk_budget = self.compute_risk_budget(current_drawdown_pct)
        if risk_budget <= 0:
            return 0

        risk_per_share = abs(entry_price - stop_loss_price)
        if risk_per_share <= 0:
            return 0

        max_risk_amount = portfolio_value * risk_budget
        shares = int(max_risk_amount / risk_per_share)

        max_position_pct = 0.05
        max_shares_by_position = int(portfolio_value * max_position_pct / entry_price)
        return min(shares, max_shares_by_position)
