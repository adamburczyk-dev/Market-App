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
    tranches: int = 1,
) -> PortfolioResult:
    """Simulate the equal-weight long-only top-quantile portfolio.

    Per session: hold the ceil(quantile * universe) symbols with the highest
    P(up); the session's gross return is their mean 1-session forward return;
    costs charge ``cost_bps`` per unit of one-way turnover (fraction of the
    book replaced vs the previous session). Degenerate inputs (no sessions)
    yield a zero result.

    ``tranches`` > 1 runs the Jegadeesh-Titman overlapping construction (T0-4):
    capital is split into `tranches` sleeves and only sleeve ``t mod tranches``
    is refreshed on session t, so a position is held for `tranches` sessions.
    Set it to the label horizon and the evaluated object finally matches the
    object the model was trained on — a 10-session label judged by a portfolio
    that turns over daily is two different bets, and the difference is paid in
    turnover.
    """
    if tranches > 1:
        return _overlapping_portfolio(
            dates, symbols, probs, next_returns, quantile, cost_bps, tranches
        )
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


@dataclass(frozen=True)
class RelativeMetrics:
    """Metrics that survive a bull market — the audit's central point (F1/T0-5).

    A long-only Sharpe answers "did the book make money", which in a window
    where 68% of names rose is a question about the market, not the model.
    These answer "did the RANKING carry information", and they do it with far
    more statistical power: SE(Sharpe) over 63 sessions is ~2.0, while the
    standard error of a mean IC over the same window is an order of magnitude
    smaller because every name in every cross-section contributes.

    - ``ic_mean`` / ``icir``  — Spearman rank correlation between prediction and
      forward return, per session; ICIR = mean/std, the information ratio of the
      signal itself.
    - ``sharpe_long_short``   — top quantile minus bottom quantile. Immune to
      the base rate by construction: if everything rises, both legs rise.
    - ``sharpe_active``       — portfolio minus the equal-weight universe.
    - ``sharpe_gross/net``    — before and after turnover costs.
    """

    ic_mean: float
    ic_std: float
    icir: float
    ic_positive_share: float
    n_cross_sections: int
    sharpe_benchmark_ew: float
    sharpe_active: float
    sharpe_long_short: float
    sharpe_gross: float
    sharpe_net: float
    cost_drag_annualized: float
    turnover_daily_mean: float


def _annualized_sharpe(returns: np.ndarray) -> float:
    if len(returns) < 2:
        return 0.0
    std = float(returns.std(ddof=1))
    return float(returns.mean() / std * math.sqrt(TRADING_DAYS)) if std > 0 else 0.0


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation without a scipy round-trip (ties averaged)."""
    if len(a) < 3:
        return 0.0
    ra, rb = _average_ranks(a), _average_ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def relative_metrics(
    dates: list[datetime],
    symbols: list[str],
    probs: np.ndarray,
    next_returns: np.ndarray,
    quantile: float = 0.2,
    cost_bps: float = 5.0,
) -> RelativeMetrics:
    """Benchmark-relative and rank-based evaluation of a set of predictions."""
    by_date: dict[datetime, list[int]] = defaultdict(list)
    for i, d in enumerate(dates):
        by_date[d].append(i)
    sessions = sorted(by_date)

    ics: list[float] = []
    long_returns: list[float] = []
    short_returns: list[float] = []
    bench_returns: list[float] = []
    gross_returns: list[float] = []
    turnovers: list[float] = []
    previous: set[str] = set()
    cost_rate = cost_bps / 10_000.0

    for session in sessions:
        rows = by_date[session]
        if len(rows) < 2:
            continue
        p = probs[rows]
        r = next_returns[rows]
        ics.append(_spearman(p, r))

        k = max(1, math.ceil(quantile * len(rows)))
        ordered = sorted(rows, key=lambda i: float(probs[i]), reverse=True)
        top, bottom = ordered[:k], ordered[-k:]
        held = {symbols[i] for i in top}

        gross = float(np.mean(next_returns[top]))
        turnover = 1.0 if not previous else len(held - previous) / len(held)
        gross_returns.append(gross)
        turnovers.append(turnover)
        long_returns.append(gross - cost_rate * turnover)
        short_returns.append(float(np.mean(next_returns[bottom])))
        bench_returns.append(float(np.mean(r)))  # equal-weight whole universe
        previous = held

    if not gross_returns:
        return RelativeMetrics(0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    ic = np.asarray(ics, dtype=float)
    net_series = np.asarray(long_returns, dtype=float)
    gross_series = np.asarray(gross_returns, dtype=float)
    bench_series = np.asarray(bench_returns, dtype=float)
    short_series = np.asarray(short_returns, dtype=float)
    turnover_mean = float(np.mean(turnovers))

    ic_std = float(ic.std(ddof=1)) if len(ic) > 1 else 0.0
    return RelativeMetrics(
        ic_mean=float(ic.mean()),
        ic_std=ic_std,
        icir=float(ic.mean() / ic_std) if ic_std > 0 else 0.0,
        ic_positive_share=float(np.mean(ic > 0)),
        n_cross_sections=len(ic),
        sharpe_benchmark_ew=_annualized_sharpe(bench_series),
        sharpe_active=_annualized_sharpe(net_series - bench_series),
        sharpe_long_short=_annualized_sharpe(net_series - short_series),
        sharpe_gross=_annualized_sharpe(gross_series),
        sharpe_net=_annualized_sharpe(net_series),
        cost_drag_annualized=float(turnover_mean * cost_rate * TRADING_DAYS),
        turnover_daily_mean=turnover_mean,
    )


def baseline_feature_ic(
    dates: list[datetime],
    feature_column: np.ndarray,
    next_returns: np.ndarray,
) -> float:
    """Mean per-session IC of a single raw feature used directly as the score.

    The rule this encodes: a model that cannot beat the rank of one feature has
    not earned the ML layer.
    """
    by_date: dict[datetime, list[int]] = defaultdict(list)
    for i, d in enumerate(dates):
        by_date[d].append(i)
    ics = [
        _spearman(feature_column[rows], next_returns[rows])
        for rows in by_date.values()
        if len(rows) >= 3
    ]
    return float(np.mean(ics)) if ics else 0.0


def _overlapping_portfolio(
    dates: list[datetime],
    symbols: list[str],
    probs: np.ndarray,
    next_returns: np.ndarray,
    quantile: float,
    cost_bps: float,
    tranches: int,
) -> PortfolioResult:
    """Overlapping-tranche construction — see top_quantile_portfolio."""
    by_date: dict[datetime, list[int]] = defaultdict(list)
    for i, d in enumerate(dates):
        by_date[d].append(i)
    sessions = sorted(by_date)
    cost_rate = cost_bps / 10_000.0

    # sleeve -> names currently held by that sleeve
    sleeves: list[set[str]] = [set() for _ in range(tranches)]
    daily_returns: list[float] = []
    turnovers: list[float] = []
    position_counts: list[int] = []

    for t, session in enumerate(sessions):
        rows = by_date[session]
        returns_by_symbol = {symbols[i]: float(next_returns[i]) for i in rows}

        active = t % tranches
        k = max(1, math.ceil(quantile * len(rows)))
        top = sorted(rows, key=lambda i: float(probs[i]), reverse=True)[:k]
        refreshed = {symbols[i] for i in top}
        # only the active sleeve trades; the rest ride their existing holdings
        replaced = len(refreshed - sleeves[active]) / len(refreshed) if refreshed else 0.0
        sleeves[active] = refreshed

        held = [name for sleeve in sleeves for name in sleeve]
        if not held:
            continue
        gross = float(np.mean([returns_by_symbol.get(name, 0.0) for name in held]))
        # the traded sleeve is 1/tranches of capital, so book-level turnover is
        # its replacement rate scaled by its weight
        turnover = replaced / tranches
        daily_returns.append(gross - cost_rate * turnover)
        turnovers.append(turnover)
        position_counts.append(len(set(held)))

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
