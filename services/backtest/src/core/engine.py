"""Position-path scoring — the one definition of what a backtest result means.

This module used to also *decide* positions, with its own trailing-momentum
rule over a price series. That made "momentum" two different strategies: the
live one on cross-sectional ranks and this one on price. A revalidation of the
live strategy therefore graded code that was never running. Deciding positions
now belongs to the registered rules (`core/rule_engine.py`); what stays here is
the arithmetic both sides must share.

Positions are decided at a bar's close and earn the *next* bar's return (no
look-ahead); position changes pay a per-turn cost in bps. Sharpe is annualized
assuming daily bars (sqrt(252)). ``first_bar`` lets the caller measure only the
out-of-sample tail while earlier bars warm the indicators up — the basis for
walk-forward revalidation (``core/walk_forward.py``).
"""

from dataclasses import dataclass

import numpy as np

TRADING_DAYS = 252


@dataclass
class BacktestParams:
    """Execution assumptions. Rule parameters are per-rule and travel separately."""

    cost_bps: float = 5.0  # cost charged on every position change (basis points)


@dataclass
class BacktestResult:
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    n_trades: int
    n_bars: int  # number of return observations actually scored

    def as_dict(self) -> dict[str, float | int]:
        return {
            "total_return": self.total_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "n_trades": self.n_trades,
            "n_bars": self.n_bars,
        }


def empty_result() -> BacktestResult:
    return BacktestResult(0.0, 0.0, 0.0, 0, 0)


def score_positions(
    prices: np.ndarray,
    position: np.ndarray,
    cost_bps: float,
    first_bar: int,
) -> BacktestResult:
    """Turn a long/flat position path into performance.

    ``position[t]`` is decided at the close of bar t and earns bar t+1's return.
    Cost is charged when a position is *established*: the trade that set
    ``held[i]`` is ``|position[i] - position[i-1]|`` (prepend 0 so the initial
    entry counts).
    """
    prices = np.asarray(prices, dtype=float)
    n = prices.size
    if n < 2 or position.size != n:
        return empty_result()

    asset_ret = np.zeros(n)
    asset_ret[1:] = prices[1:] / prices[:-1] - 1.0

    held = position[:-1]  # length n-1, index i ↔ bar t=i+1
    gross = held * asset_ret[1:]
    trade = np.abs(np.diff(position, prepend=0.0))[:-1]  # aligned to held
    strat_ret = gross - trade * (cost_bps / 10_000.0)

    first = max(first_bar, 1)
    scored = strat_ret[first - 1 :]  # bar t maps to index t-1
    if scored.size == 0:
        return empty_result()

    n_trades = int(np.count_nonzero(trade[first - 1 :] > 0))
    equity = np.cumprod(1.0 + scored)
    total_return = float(equity[-1] - 1.0)

    std = float(scored.std(ddof=1)) if scored.size > 1 else 0.0
    sharpe = float(scored.mean() / std * np.sqrt(TRADING_DAYS)) if std > 0 else 0.0

    running_max = np.maximum.accumulate(equity)
    max_dd = float(np.max((running_max - equity) / running_max)) if equity.size else 0.0

    return BacktestResult(
        total_return=total_return,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        n_trades=n_trades,
        n_bars=int(scored.size),
    )
