"""Portfolio circuit breaker — armed 24/7.

Escalating levels (worst wins):
- BLACK: drawdown > flatten threshold → flatten all, human restart required.
- RED:   daily loss > halt threshold → halt new trading for the day.
- YELLOW: drawdown > warn threshold → reduce risk.

Tripped (RED/BLACK) blocks new orders. Levels auto-clear when conditions
improve (a real system would require a manual reset out of BLACK — TODO).
"""

from dataclasses import dataclass

from trading_common.events import CircuitBreakerLevel


@dataclass
class BreakerResult:
    level: CircuitBreakerLevel | None
    changed: bool
    trigger_metric: str
    current_value: float
    threshold: float
    action: str


class CircuitBreaker:
    def __init__(
        self,
        drawdown_warn_pct: float = 0.08,
        daily_loss_halt_pct: float = 0.05,
        drawdown_flatten_pct: float = 0.15,
    ) -> None:
        self._warn_dd = drawdown_warn_pct
        self._halt_daily = daily_loss_halt_pct
        self._flatten_dd = drawdown_flatten_pct
        self._level: CircuitBreakerLevel | None = None

    @property
    def level(self) -> CircuitBreakerLevel | None:
        return self._level

    @property
    def is_tripped(self) -> bool:
        return self._level in (CircuitBreakerLevel.RED, CircuitBreakerLevel.BLACK)

    def evaluate(self, drawdown_pct: float, daily_loss_pct: float) -> BreakerResult:
        dd = abs(drawdown_pct)
        dl = abs(daily_loss_pct)
        if dd >= self._flatten_dd:
            return self._set(
                CircuitBreakerLevel.BLACK, "drawdown", dd, self._flatten_dd, "flatten_all"
            )
        if dl >= self._halt_daily:
            return self._set(
                CircuitBreakerLevel.RED, "daily_loss", dl, self._halt_daily, "halt_trading"
            )
        if dd >= self._warn_dd:
            return self._set(
                CircuitBreakerLevel.YELLOW, "drawdown", dd, self._warn_dd, "reduce_risk"
            )
        return self._set(None, "none", dd, self._warn_dd, "none")

    def _set(
        self,
        level: CircuitBreakerLevel | None,
        metric: str,
        current: float,
        threshold: float,
        action: str,
    ) -> BreakerResult:
        changed = level != self._level
        self._level = level
        return BreakerResult(level, changed, metric, current, threshold, action)
