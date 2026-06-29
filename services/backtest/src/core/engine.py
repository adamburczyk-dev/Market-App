"""Vectorized time-series momentum backtest engine.

A deliberately simple long/flat momentum rule over a single symbol's closes:
go long when trailing momentum is positive, otherwise hold cash. Positions are
decided at a bar's close and earn the *next* bar's return (no look-ahead);
position changes pay a per-turn cost in bps.

Sharpe is annualized assuming daily bars (sqrt(252)). ``start_index`` lets the
caller measure performance over only the out-of-sample tail while still using
the earlier bars to warm up the momentum lookback — the basis for walk-forward
revalidation (``core/walk_forward.py``).
"""

from dataclasses import dataclass

import numpy as np

TRADING_DAYS = 252


@dataclass
class BacktestParams:
    lookback: int = 20  # bars used for the trailing-momentum signal
    entry_momentum: float = 0.0  # go long when momentum > this threshold
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


def _empty() -> BacktestResult:
    return BacktestResult(0.0, 0.0, 0.0, 0, 0)


def run_backtest(
    closes: list[float] | np.ndarray,
    params: BacktestParams,
    start_index: int | None = None,
) -> BacktestResult:
    """Run the momentum long/flat backtest over ``closes``.

    ``start_index`` (default: score from the first bar that has both a lookback
    window and a prior position) restricts the scored window to ``[start_index,
    end]`` so the out-of-sample tail can be measured in isolation.
    """
    prices = np.asarray(closes, dtype=float)
    n = prices.size
    # Need at least lookback+2 bars: one to form momentum, one prior position, one return.
    if n < params.lookback + 2:
        return _empty()

    # Trailing momentum: price[t] / price[t - lookback] - 1, defined for t >= lookback.
    lb = params.lookback
    momentum = np.full(n, np.nan)
    momentum[lb:] = prices[lb:] / prices[:-lb] - 1.0

    # Position decided at close of bar t (long/flat); NaN momentum → flat.
    position = np.where(momentum > params.entry_momentum, 1.0, 0.0)
    position[:lb] = 0.0

    # Asset simple returns, aligned so r[t] is the return from t-1 to t.
    asset_ret = np.zeros(n)
    asset_ret[1:] = prices[1:] / prices[:-1] - 1.0

    # No look-ahead: the position held into bar t is position[t-1].
    held = position[:-1]  # length n-1, index i ↔ bar t=i+1
    gross = held * asset_ret[1:]
    # Cost is charged when a position is *established*: the trade that set held[i]
    # is |position[i] - position[i-1]| (prepend 0 so the initial entry counts).
    trade = np.abs(np.diff(position, prepend=0.0))[:-1]  # aligned to held
    strat_ret = gross - trade * (params.cost_bps / 10_000.0)

    # Scored window: first scorable bar is t=lb+1 (first bar earning under a real position).
    first = lb + 1 if start_index is None else max(start_index, lb + 1)
    # bar t maps to index t-1 in the length-(n-1) arrays.
    scored = strat_ret[first - 1 :]
    if scored.size == 0:
        return _empty()

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
