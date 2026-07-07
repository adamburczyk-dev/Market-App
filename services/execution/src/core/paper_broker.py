"""In-memory paper broker — simulates fills and tracks portfolio state.

Mark-to-fill: positions are valued at their last fill/mark price. Tracks peak
equity (for drawdown) and the day's starting equity (for daily loss); the day
baseline rolls over on the first fill/mark of a new day, so "daily loss" is
truly daily and the RED halt clears next day (R2). Fills are idempotent by
order_id, so a NATS redelivery cannot double-fill (R3). Long positions carry
their protective levels so the service can exit them on re-marks (R5).
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime


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
    stop_loss: float | None = None
    take_profit: float | None = None


def _today() -> date:
    return datetime.now(UTC).date()


@dataclass
class PaperBroker:
    initial_cash: float = 100_000.0
    slippage_bps: float = 0.0
    clock: Callable[[], date] = _today
    _cash: float = field(init=False)
    _positions: dict[str, Position] = field(init=False, default_factory=dict)
    _peak_equity: float = field(init=False)
    _day_start_equity: float = field(init=False)
    _day_start_date: date = field(init=False)
    _processed_orders: set[str] = field(init=False, default_factory=set)

    def __post_init__(self) -> None:
        self._cash = self.initial_cash
        self._peak_equity = self.initial_cash
        self._day_start_equity = self.initial_cash
        self._day_start_date = self.clock()

    def _roll_day_if_needed(self) -> None:
        """The first event of a new day resets the daily-loss baseline."""
        today = self.clock()
        if today != self._day_start_date:
            self._day_start_date = today
            self._day_start_equity = self.equity

    def is_processed(self, order_id: str) -> bool:
        return order_id in self._processed_orders

    def fill(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> Fill | None:
        """Fill an order; idempotent by ``order_id`` (a duplicate returns None)."""
        if order_id in self._processed_orders:
            return None
        self._roll_day_if_needed()
        fill_price = self._apply_slippage(side, price)
        pos = self._positions.setdefault(symbol, Position())
        if side == "BUY":
            self._cash -= quantity * fill_price
            pos.quantity += quantity
            # The latest BUY defines the position's protective levels
            # (paper simplification — no per-lot level tracking).
            if stop_loss is not None:
                pos.stop_loss = stop_loss
            if take_profit is not None:
                pos.take_profit = take_profit
        else:  # SELL
            self._cash += quantity * fill_price
            pos.quantity -= quantity
            if pos.quantity <= 0:
                pos.stop_loss = None
                pos.take_profit = None
        pos.last_price = fill_price
        self._peak_equity = max(self._peak_equity, self.equity)
        self._processed_orders.add(order_id)
        return Fill(order_id, symbol, side, quantity, fill_price)

    def _apply_slippage(self, side: str, price: float) -> float:
        slip = self.slippage_bps / 10_000.0
        return price * (1 + slip) if side == "BUY" else price * (1 - slip)

    def mark(self, symbol: str, price: float) -> None:
        """Re-mark an open position to a new market price (unrealized P&L)."""
        pos = self._positions.get(symbol)
        if pos is None or pos.quantity == 0:
            return
        self._roll_day_if_needed()
        pos.last_price = price
        self._peak_equity = max(self._peak_equity, self.equity)

    def position_qty(self, symbol: str) -> float:
        pos = self._positions.get(symbol)
        return pos.quantity if pos is not None else 0.0

    def protective_trigger(self, symbol: str) -> str | None:
        """Return "stop_loss"/"take_profit" when the marked price breaches a level."""
        pos = self._positions.get(symbol)
        if pos is None or pos.quantity <= 0:
            return None
        if pos.stop_loss is not None and pos.last_price <= pos.stop_loss:
            return "stop_loss"
        if pos.take_profit is not None and pos.last_price >= pos.take_profit:
            return "take_profit"
        return None

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
            symbol: {
                "quantity": p.quantity,
                "last_price": p.last_price,
                "stop_loss": p.stop_loss,
                "take_profit": p.take_profit,
            }
            for symbol, p in self._positions.items()
            if p.quantity != 0
        }

    def snapshot(self) -> dict:
        """Serializable broker state for persistence."""
        return {
            "cash": self._cash,
            "peak_equity": self._peak_equity,
            "day_start_equity": self._day_start_equity,
            "day_start_date": self._day_start_date.isoformat(),
            "processed_orders": sorted(self._processed_orders),
            "positions": {
                symbol: {
                    "quantity": p.quantity,
                    "last_price": p.last_price,
                    "stop_loss": p.stop_loss,
                    "take_profit": p.take_profit,
                }
                for symbol, p in self._positions.items()
            },
        }

    def restore(self, snapshot: dict) -> None:
        """Re-apply a persisted snapshot (tolerant of older snapshot layouts)."""
        self._cash = snapshot["cash"]
        self._peak_equity = snapshot["peak_equity"]
        self._day_start_equity = snapshot["day_start_equity"]
        raw_date = snapshot.get("day_start_date")
        self._day_start_date = date.fromisoformat(raw_date) if raw_date else self.clock()
        self._processed_orders = set(snapshot.get("processed_orders", []))
        self._positions = {
            symbol: Position(
                quantity=p["quantity"],
                last_price=p["last_price"],
                stop_loss=p.get("stop_loss"),
                take_profit=p.get("take_profit"),
            )
            for symbol, p in snapshot.get("positions", {}).items()
        }
