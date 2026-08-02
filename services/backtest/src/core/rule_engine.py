"""Backtest a REGISTERED strategy rule — the same object the service trades.

Until now this service reimplemented momentum on a price series while the live
path ran it on cross-sectional ranks, and `run_backtest` did not even read the
`strategy_name` it was given: any name produced the same numbers. A weekly
revalidation therefore graded a strategy that was not the one in production, and
nothing said so.

Here the rule comes from `trading_common.strategies`, so there is exactly one
definition of each rule and backtest evaluates that one.

**Two limits, both deliberate and both named rather than papered over:**

1. A rule declaring `required_ranks` reads cross-sectional percentiles, which
   only exist relative to a universe. A single-symbol backtest cannot produce
   them, so this refuses instead of substituting a price-based proxy — that
   substitution IS the defect being removed. A cross-sectional engine with
   `1/h` tranches is the open decision D7.
2. Positions are long/flat and rebalanced daily, matching the existing engine
   and the live long-only path (R4). BUY enters, SELL exits, HOLD carries the
   previous position — a rule that fires rarely must not be read as "flat",
   which is what recomputing the position from scratch each bar would do.
"""

import numpy as np
import structlog
from trading_common.features import FEATURE_LOOKBACK, compute_feature_vector
from trading_common.schemas import OHLCVBar, Signal
from trading_common.strategies import StrategyRule

from src.core.engine import BacktestResult, score_positions

logger = structlog.get_logger()


class CrossSectionalRuleError(ValueError):
    """Raised for a rule that cannot be evaluated one symbol at a time."""


def rule_positions(
    bars: list[OHLCVBar],
    rule: StrategyRule,
    params: dict[str, float] | None = None,
) -> np.ndarray:
    """Long/flat position decided at each bar's close by `rule`.

    The feature vector is computed from the SAME trailing window the serving
    path uses (`FEATURE_LOOKBACK`), so the rule sees the same numbers here as it
    will live. A shorter window here would make the backtest grade a different
    function under the same name — the exact class of drift this module exists
    to remove.
    """
    if rule.required_ranks:
        raise CrossSectionalRuleError(
            f"strategy {rule.name} reads cross-sectional ranks {sorted(rule.required_ranks)}, "
            "which do not exist for a single symbol — a universe backtest is required"
        )

    position = np.zeros(len(bars))
    current = 0.0
    for t in range(len(bars)):
        window = bars[max(0, t - FEATURE_LOOKBACK + 1) : t + 1]
        features = compute_feature_vector(window).features
        decision = rule.generate({}, features, params)
        if decision.signal == Signal.BUY:
            current = 1.0
        elif decision.signal == Signal.SELL:
            current = 0.0
        # HOLD: carry the previous position.
        position[t] = current
    return position


def run_rule_backtest(
    bars: list[OHLCVBar],
    rule: StrategyRule,
    prices: np.ndarray,
    cost_bps: float = 5.0,
    params: dict[str, float] | None = None,
    start_index: int | None = None,
) -> BacktestResult:
    """Score `rule` over `bars`. `prices` is the return series (adjusted closes).

    The scored window starts at `start_index` (default: the first bar the rule
    could act on) so a walk-forward caller can measure only the out-of-sample
    tail while the earlier bars warm the indicators up.
    """
    if len(bars) < 2:
        return BacktestResult(0.0, 0.0, 0.0, 0, 0)
    position = rule_positions(bars, rule, params)
    first = 1 if start_index is None else max(start_index, 1)
    result = score_positions(prices, position, cost_bps, first)
    logger.info(
        "Rule backtest",
        strategy=rule.name,
        bars=len(bars),
        scored=result.n_bars,
        trades=result.n_trades,
        sharpe=round(result.sharpe_ratio, 4),
    )
    return result
