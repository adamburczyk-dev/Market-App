"""Testy circuit breakera."""

from trading_common.events import CircuitBreakerLevel

from src.core.circuit_breaker import CircuitBreaker


def test_no_trip_when_calm():
    cb = CircuitBreaker()
    result = cb.evaluate(drawdown_pct=0.02, daily_loss_pct=0.01)
    assert result.level is None
    assert cb.is_tripped is False


def test_yellow_on_drawdown_warning():
    cb = CircuitBreaker()
    result = cb.evaluate(drawdown_pct=0.10, daily_loss_pct=0.0)  # > 8%
    assert result.level == CircuitBreakerLevel.YELLOW
    assert result.changed is True
    assert cb.is_tripped is False  # yellow does not block


def test_red_halts_on_daily_loss():
    cb = CircuitBreaker()
    result = cb.evaluate(drawdown_pct=0.0, daily_loss_pct=0.06)  # > 5%
    assert result.level == CircuitBreakerLevel.RED
    assert result.action == "halt_trading"
    assert cb.is_tripped is True


def test_black_flattens_on_deep_drawdown():
    cb = CircuitBreaker()
    result = cb.evaluate(drawdown_pct=0.16, daily_loss_pct=0.0)  # > 15%
    assert result.level == CircuitBreakerLevel.BLACK
    assert result.action == "flatten_all"
    assert cb.is_tripped is True


def test_drawdown_beats_daily_loss():
    cb = CircuitBreaker()
    result = cb.evaluate(drawdown_pct=0.16, daily_loss_pct=0.06)
    assert result.level == CircuitBreakerLevel.BLACK  # worst wins


def test_changed_flag_only_on_transition():
    cb = CircuitBreaker()
    first = cb.evaluate(drawdown_pct=0.10, daily_loss_pct=0.0)
    second = cb.evaluate(drawdown_pct=0.11, daily_loss_pct=0.0)
    assert first.changed is True
    assert second.changed is False  # still YELLOW


def test_auto_clears_when_conditions_improve():
    cb = CircuitBreaker()
    cb.evaluate(drawdown_pct=0.10, daily_loss_pct=0.0)  # YELLOW
    result = cb.evaluate(drawdown_pct=0.0, daily_loss_pct=0.0)
    assert result.level is None
    assert result.changed is True
    assert cb.is_tripped is False
