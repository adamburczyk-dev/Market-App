"""OOS evaluation — rank diagnostics + the decision metric (plan §6).

AUC and Brier are diagnostics; the DECISION metric is the cost-adjusted
Sharpe of a daily-rebalanced, equal-weight, long-only top-quantile portfolio
built from out-of-sample predictions — that is what the activation gate
(OOS Sharpe > 0.5) reads. Pure numpy, no torch.
"""

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

import numpy as np

TRADING_DAYS = 252


def auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """ROC AUC via the Mann-Whitney U rank statistic (tie-aware).

    Returns 0.5 for degenerate inputs (single class) — the uninformative value.
    """
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_scores = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0  # 1-based average rank
        i = j + 1
    rank_sum_pos = float(ranks[y == 1].sum())
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def brier(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Mean squared error of the probability forecast (lower is better)."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probs, dtype=float)
    return float(np.mean((p - y) ** 2))


@dataclass(frozen=True)
class PortfolioResult:
    sharpe: float
    mean_daily_return: float  # net of costs
    n_sessions: int
    avg_positions: float
    avg_turnover: float  # fraction of the book replaced per session


def top_quantile_portfolio(
    dates: list[datetime],
    symbols: list[str],
    probs: np.ndarray,
    next_returns: np.ndarray,
    quantile: float = 0.2,
    cost_bps: float = 5.0,
) -> PortfolioResult:
    """Simulate the equal-weight long-only top-quantile portfolio.

    Per session: hold the ceil(quantile * universe) symbols with the highest
    P(up); the session's gross return is their mean 1-session forward return;
    costs charge ``cost_bps`` per unit of one-way turnover (fraction of the
    book replaced vs the previous session). Degenerate inputs (no sessions)
    yield a zero result.
    """
    by_date: dict[datetime, list[int]] = defaultdict(list)
    for i, d in enumerate(dates):
        by_date[d].append(i)

    sessions = sorted(by_date)
    daily_returns: list[float] = []
    turnovers: list[float] = []
    position_counts: list[int] = []
    previous: set[str] = set()
    cost_rate = cost_bps / 10_000.0

    for session in sessions:
        rows = by_date[session]
        k = max(1, math.ceil(quantile * len(rows)))
        top = sorted(rows, key=lambda i: float(probs[i]), reverse=True)[:k]
        held = {symbols[i] for i in top}
        gross = float(np.mean(next_returns[top]))
        turnover = 1.0 if not previous else len(held - previous) / len(held)
        daily_returns.append(gross - cost_rate * turnover)
        turnovers.append(turnover)
        position_counts.append(len(held))
        previous = held

    if not daily_returns:
        return PortfolioResult(0.0, 0.0, 0, 0.0, 0.0)

    returns = np.asarray(daily_returns, dtype=float)
    std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    sharpe = float(returns.mean() / std * math.sqrt(TRADING_DAYS)) if std > 0 else 0.0
    return PortfolioResult(
        sharpe=sharpe,
        mean_daily_return=float(returns.mean()),
        n_sessions=len(returns),
        avg_positions=float(np.mean(position_counts)),
        avg_turnover=float(np.mean(turnovers)),
    )
