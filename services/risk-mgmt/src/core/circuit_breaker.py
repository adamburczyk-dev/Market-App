"""Portfolio circuit breaker — armed 24/7.

Escalating levels (worst wins):

- **BLACK**: drawdown > flatten threshold → flatten all, and **stay tripped
  until a human resets it**.
- **RED**: daily loss > halt threshold → halt new trading **for the rest of the
  session**.
- **YELLOW**: drawdown > warn threshold → reduce risk (advisory, self-clearing).

Tripped (RED/BLACK) blocks new orders, never liquidations — closing a position
is itself an order, and refusing it is the opposite of what the breaker is for.

**Why two of these are sticky rather than metric-driven.** The rules are written
in terms of time and human authority, not in terms of the current reading:

    "Daily loss > 5% → automatic trading halt UNTIL NEXT DAY"
    "Drawdown > 15% → flatten all positions, REQUIRE HUMAN RESTART"

A breaker that simply reports the present metric satisfies neither. It lifts the
daily halt the moment an intraday bounce takes the loss back under 5% — on the
same day, which is precisely what "until next day" forbids — and it forgets a
BLACK the moment drawdown recovers, so the system resumes trading after a
catastrophic loss with nobody having looked at it. Both failures share a shape:
they can only happen after something already went badly wrong, and they make the
recovery look like permission.

So BLACK **latches** (cleared only by `reset()`) and RED **holds for its
session** (cleared by the date rolling over). Both pieces of state are persisted,
because otherwise restarting the container would be an accidental human reset —
the easiest possible way to bypass the rule, and one nobody would notice.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime

import structlog
from trading_common.events import CircuitBreakerLevel

logger = structlog.get_logger()


@dataclass
class BreakerResult:
    level: CircuitBreakerLevel | None
    changed: bool
    trigger_metric: str
    current_value: float
    threshold: float
    action: str


@dataclass(frozen=True)
class ResetResult:
    """Outcome of a human reset attempt."""

    cleared: bool
    level: CircuitBreakerLevel | None  # level AFTER the attempt
    reason: str


class CircuitBreaker:
    def __init__(
        self,
        drawdown_warn_pct: float = 0.08,
        daily_loss_halt_pct: float = 0.05,
        drawdown_flatten_pct: float = 0.15,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._warn_dd = drawdown_warn_pct
        self._halt_daily = daily_loss_halt_pct
        self._flatten_dd = drawdown_flatten_pct
        self._clock = clock or (lambda: datetime.now(UTC))
        self._level: CircuitBreakerLevel | None = None
        # Set when drawdown breached the flatten threshold. Only reset() clears
        # it — not a recovery, and not a restart.
        self._latched_black = False
        # The session a daily-loss halt belongs to. The halt stands for the rest
        # of that session regardless of what the metric does afterwards.
        self._halted_session: date | None = None

    @property
    def level(self) -> CircuitBreakerLevel | None:
        return self._level

    @property
    def is_tripped(self) -> bool:
        return self._level in (CircuitBreakerLevel.RED, CircuitBreakerLevel.BLACK)

    @property
    def latched(self) -> bool:
        """BLACK is being held open pending a human reset."""
        return self._latched_black

    @property
    def halted_session(self) -> date | None:
        return self._halted_session

    def evaluate(self, drawdown_pct: float, daily_loss_pct: float) -> BreakerResult:
        dd = abs(drawdown_pct)
        dl = abs(daily_loss_pct)
        today = self._clock().astimezone(UTC).date()

        if dd >= self._flatten_dd:
            self._latched_black = True
        if dl >= self._halt_daily:
            self._halted_session = today

        # A latched BLACK outranks everything and ignores the current reading:
        # the drawdown recovering is not the event that clears it.
        if self._latched_black:
            return self._set(
                CircuitBreakerLevel.BLACK, "drawdown", dd, self._flatten_dd, "flatten_all"
            )
        if self._halted_session == today:
            return self._set(
                CircuitBreakerLevel.RED, "daily_loss", dl, self._halt_daily, "halt_trading"
            )
        if dd >= self._warn_dd:
            return self._set(
                CircuitBreakerLevel.YELLOW, "drawdown", dd, self._warn_dd, "reduce_risk"
            )
        return self._set(None, "none", dd, self._warn_dd, "none")

    def reset(self, drawdown_pct: float, daily_loss_pct: float) -> ResetResult:
        """Human clears a latched BLACK. Refused while the breach still stands.

        Refusing is the point: clearing a latch while drawdown is still past the
        flatten threshold would re-trip on the next portfolio update anyway, but
        in between it would let new orders through at the worst possible moment.
        The operator has to bring the book back inside the limit first — the
        reset acknowledges a recovery, it does not perform one.
        """
        dd = abs(drawdown_pct)
        if not self._latched_black:
            return ResetResult(False, self._level, "nothing latched")
        if dd >= self._flatten_dd:
            return ResetResult(
                False,
                self._level,
                f"drawdown {dd:.1%} is still at or past the {self._flatten_dd:.0%} "
                "flatten threshold — reduce exposure first",
            )
        self._latched_black = False
        logger.warning("Circuit breaker manually reset", drawdown_pct=dd)
        result = self.evaluate(dd, daily_loss_pct)
        return ResetResult(True, result.level, "reset accepted")

    def snapshot(self) -> dict:
        """Latch state for persistence — a restart must not act as a reset."""
        return {
            "latched_black": self._latched_black,
            "halted_session": self._halted_session.isoformat() if self._halted_session else None,
        }

    def restore(self, snapshot: dict | None) -> None:
        if not snapshot:
            return
        self._latched_black = bool(snapshot.get("latched_black", False))
        halted = snapshot.get("halted_session")
        self._halted_session = date.fromisoformat(halted) if halted else None

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
