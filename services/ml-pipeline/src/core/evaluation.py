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


@dataclass(frozen=True)
class SelectionDiagnostics:
    """Why a fold scored what it scored — read alongside the Sharpe.

    Sharpe on a short window is noisy; these say whether the model actually
    discriminates. ``lift`` is the edge that the portfolio monetizes: how much
    more often the SELECTED rows (the same per-session top quantile the
    portfolio holds) went up than the population. Zero lift with a high Sharpe
    means luck, not signal. The prediction spread catches the other failure
    mode — a collapsed model that outputs one constant probability.
    """

    base_rate: float  # share of label==1 over all rows
    selected_hit_rate: float  # share of label==1 among top-quantile picks
    lift: float  # selected_hit_rate − base_rate
    pred_mean: float
    pred_std: float  # ≈0 → degenerate model, no ranking information
    pred_p10: float
    pred_p90: float


def selection_diagnostics(
    dates: list[datetime],
    y_true: np.ndarray,
    probs: np.ndarray,
    quantile: float = 0.2,
) -> SelectionDiagnostics:
    """Label statistics of the per-session top-quantile selection."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probs, dtype=float)
    by_date: dict[datetime, list[int]] = defaultdict(list)
    for i, d in enumerate(dates):
        by_date[d].append(i)

    selected: list[int] = []
    for rows in by_date.values():
        k = max(1, math.ceil(quantile * len(rows)))
        selected.extend(sorted(rows, key=lambda i: float(p[i]), reverse=True)[:k])

    base_rate = float(y.mean()) if len(y) else 0.0
    hit_rate = float(y[selected].mean()) if selected else 0.0
    return SelectionDiagnostics(
        base_rate=base_rate,
        selected_hit_rate=hit_rate,
        lift=hit_rate - base_rate,
        pred_mean=float(p.mean()) if len(p) else 0.0,
        pred_std=float(p.std()) if len(p) else 0.0,
        pred_p10=float(np.quantile(p, 0.1)) if len(p) else 0.0,
        pred_p90=float(np.quantile(p, 0.9)) if len(p) else 0.0,
    )


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


@dataclass(frozen=True)
class EffectiveSampleSize:
    """How much independent information the dataset really carries.

    48 827 rows is a nominal count. Labels span `horizon` sessions, so
    consecutive rows for one symbol overlap almost entirely, and large caps
    move together, so names are not independent draws either. Both shrinkages
    are measured here rather than assumed: the time axis divides by the
    horizon, the cross-section divides by the average pairwise correlation via
    the standard effective-breadth formula N / (1 + (N-1)·rho).

    Read it against the feature count: below ~50 observations per feature,
    financial datasets do not support reliable inference.
    """

    n_samples: int
    n_sessions: int
    n_symbols: int
    avg_pairwise_correlation: float
    n_symbols_effective: float
    n_independent_periods: float
    n_effective_samples: float


def effective_sample_size(
    dates: list[datetime],
    symbols: list[str],
    next_returns: np.ndarray,
    horizon: int,
) -> EffectiveSampleSize:
    """Measure independent information content (T0-3)."""
    n_samples = len(dates)
    sessions = sorted(set(dates))
    names = sorted(set(symbols))
    if n_samples == 0 or not sessions:
        return EffectiveSampleSize(0, 0, 0, 0.0, 0.0, 0.0, 0.0)

    # pivot returns into (session x symbol); gaps stay NaN
    row_of = {d: i for i, d in enumerate(sessions)}
    col_of = {s: j for j, s in enumerate(names)}
    matrix = np.full((len(sessions), len(names)), np.nan)
    for date, symbol, ret in zip(dates, symbols, next_returns, strict=True):
        matrix[row_of[date], col_of[symbol]] = ret

    correlations: list[float] = []
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            pair = matrix[:, [a, b]]
            usable = pair[~np.isnan(pair).any(axis=1)]
            if len(usable) > 30 and usable[:, 0].std() > 0 and usable[:, 1].std() > 0:
                correlations.append(float(np.corrcoef(usable[:, 0], usable[:, 1])[0, 1]))
    rho = float(np.mean(correlations)) if correlations else 0.0

    n = len(names)
    effective_names = n / (1.0 + (n - 1) * rho) if n > 1 and rho > 0 else float(n)
    independent_periods = len(sessions) / horizon if horizon > 0 else float(len(sessions))
    return EffectiveSampleSize(
        n_samples=n_samples,
        n_sessions=len(sessions),
        n_symbols=n,
        avg_pairwise_correlation=round(rho, 4),
        n_symbols_effective=round(effective_names, 2),
        n_independent_periods=round(independent_periods, 1),
        n_effective_samples=round(independent_periods * effective_names, 1),
    )
