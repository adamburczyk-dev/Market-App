"""In-memory paper broker — simulates fills and tracks portfolio state.

Mark-to-fill: positions are valued at their last fill price (no live marks yet).
Tracks peak equity (for drawdown) and the day's starting equity (for daily loss)
so it can feed risk-mgmt the metrics that drive sizing and the circuit breaker.
"""

from dataclasses import dataclass, field


@dataclass
class Fill:
    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float


@dataclass
class Position:
    quantity: float = 0.0
    last_price: float = 0.0


@dataclass
class PaperBroker:
    initial_cash: float = 100_000.0
    slippage_bps: float = 0.0
    _cash: float = field(init=False)
    _positions: dict[str, Position] = field(init=False, default_factory=dict)
    _peak_equity: float = field(init=False)
    _day_start_equity: float = field(init=False)

    def __post_init__(self) -> None:
        self._cash = self.initial_cash
        self._peak_equity = self.initial_cash
        self._day_start_equity = self.initial_cash

    def fill(self, order_id: str, symbol: str, side: str, quantity: float, price: float) -> Fill:
        fill_price = self._apply_slippage(side, price)
        pos = self._positions.setdefault(symbol, Position())
        if side == "BUY":
            self._cash -= quantity * fill_price
            pos.quantity += quantity
        else:  # SELL
            self._cash += quantity * fill_price
            pos.quantity -= quantity
        pos.last_price = fill_price
        self._peak_equity = max(self._peak_equity, self.equity)
        return Fill(order_id, symbol, side, quantity, fill_price)

    def _apply_slippage(self, side: str, price: float) -> float:
        slip = self.slippage_bps / 10_000.0
        return price * (1 + slip) if side == "BUY" else price * (1 - slip)

    def mark(self, symbol: str, price: float) -> None:
        """Re-mark an open position to a new market price (unrealized P&L)."""
        pos = self._positions.get(symbol)
        if pos is None or pos.quantity == 0:
            return
        pos.last_price = price
        self._peak_equity = max(self._peak_equity, self.equity)

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def positions_value(self) -> float:
        return sum(p.quantity * p.last_price for p in self._positions.values())

    @property
    def gross_exposure_value(self) -> float:
        return sum(abs(p.quantity) * p.last_price for p in self._positions.values())

    @property
    def equity(self) -> float:
        return self._cash + self.positions_value

    def metrics(self) -> dict:
        """Portfolio metrics for risk-mgmt (exposure / drawdown / daily loss)."""
        equity = self.equity
        drawdown = (
            (self._peak_equity - equity) / self._peak_equity if self._peak_equity > 0 else 0.0
        )
        daily_loss = (
            (self._day_start_equity - equity) / self._day_start_equity
            if self._day_start_equity > 0
            else 0.0
        )
        exposure = self.gross_exposure_value / equity if equity > 0 else 0.0
        return {
            "value": equity,
            "exposure_pct": exposure,
            "drawdown_pct": max(0.0, drawdown),
            "daily_loss_pct": max(0.0, daily_loss),
        }

    def positions(self) -> dict:
        return {
            symbol: {"quantity": p.quantity, "last_price": p.last_price}
            for symbol, p in self._positions.items()
            if p.quantity != 0
        }

    def snapshot(self) -> dict:
        """Serializable broker state for persistence (cash, positions, equity highs)."""
        return {
            "cash": self._cash,
            "peak_equity": self._peak_equity,
            "day_start_equity": self._day_start_equity,
            "positions": {
                symbol: {"quantity": p.quantity, "last_price": p.last_price}
                for symbol, p in self._positions.items()
            },
        }

    def restore(self, snapshot: dict) -> None:
        """Re-apply a persisted snapshot (overwrites current in-memory state)."""
        self._cash = snapshot["cash"]
        self._peak_equity = snapshot["peak_equity"]
        self._day_start_equity = snapshot["day_start_equity"]
        self._positions = {
            symbol: Position(quantity=p["quantity"], last_price=p["last_price"])
            for symbol, p in snapshot.get("positions", {}).items()
        }
