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
    # The realized net daily series. Kept as a tuple (comparable, hashable —
    # an ndarray field would break equality) because the deflated Sharpe needs
    # the higher moments, not just the ratio: a Sharpe built on a few lucky
    # skewed sessions deflates far more than the same Sharpe from a steady one.
    returns: tuple[float, ...] = ()


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
        returns=tuple(float(r) for r in returns),
    )


EULER_MASCHERONI = 0.5772156649015329


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _normal_ppf(p: float) -> float:
    """Inverse normal CDF (Acklam's rational approximation, ~1e-9 accurate)."""
    p = min(max(p, 1e-12), 1 - 1e-12)
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )


def expected_max_sharpe(n_trials: int, sharpe_std: float = 1.0) -> float:
    """Expected MAXIMUM Sharpe from `n_trials` strategies with zero true edge.

    Trying many configurations and keeping the best one produces a positive
    Sharpe with certainty; this is the bar that result has to clear before it
    means anything (Bailey & López de Prado).

    A single trial has no selection to correct for — the deflated ratio then
    reduces to the probabilistic Sharpe against a zero benchmark.
    """
    n = int(n_trials)
    if n <= 1:
        return 0.0
    return sharpe_std * (
        (1 - EULER_MASCHERONI) * _normal_ppf(1 - 1.0 / n)
        + EULER_MASCHERONI * _normal_ppf(1 - 1.0 / (n * math.e))
    )


def deflated_sharpe_ratio(
    returns: tuple[float, ...] | np.ndarray,
    n_trials: int = 1,
    benchmark_sharpe: float | None = None,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Probability that the true Sharpe exceeds the selection-adjusted benchmark.

    Corrects the two things a raw Sharpe hides: how many attempts it took to
    find this one (the benchmark defaults to the expected maximum under the
    null for ``n_trials``), and the shape of the return distribution — negative
    skew and fat tails make the same ratio less trustworthy. Returns a
    probability in [0, 1]; ~0.95 is the usual bar.
    """
    r = np.asarray(returns, dtype=float)
    n = len(r)
    if n < 8:
        return 0.0
    std = float(r.std(ddof=1))
    if std <= 0:
        return 0.0
    sr = float(r.mean()) / std  # per-period, matching the moments below
    centered = (r - r.mean()) / std
    skew = float(np.mean(centered**3))
    kurtosis = float(np.mean(centered**4))
    sr0 = (
        benchmark_sharpe / math.sqrt(periods_per_year)
        if benchmark_sharpe is not None
        else expected_max_sharpe(n_trials) / math.sqrt(n - 1)
    )
    denominator = 1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr**2
    if denominator <= 0:
        return 0.0
    return _normal_cdf((sr - sr0) * math.sqrt(n - 1) / math.sqrt(denominator))


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


def _tranche_holdings(
    sessions: list[datetime],
    by_date: dict[datetime, list[int]],
    symbols: list[str],
    probs: np.ndarray,
    quantile: float,
    tranches: int,
    *,
    highest: bool = True,
) -> list[tuple[list[str], float]]:
    """Per-session (held names, book turnover) under the tranche rule.

    ``tranches == 1`` is the daily-rebalanced book; larger values split capital
    into sleeves and refresh sleeve ``t mod tranches`` only, so a position is
    held for ``tranches`` sessions (Jegadeesh-Titman).
    """
    sleeves: list[set[str]] = [set() for _ in range(tranches)]
    out: list[tuple[list[str], float]] = []
    for t, session in enumerate(sessions):
        rows = by_date[session]
        k = max(1, math.ceil(quantile * len(rows)))
        ordered = sorted(rows, key=lambda i: float(probs[i]), reverse=highest)
        refreshed = {symbols[i] for i in ordered[:k]}
        active = t % tranches
        replaced = len(refreshed - sleeves[active]) / len(refreshed) if refreshed else 0.0
        sleeves[active] = refreshed
        out.append(([name for sleeve in sleeves for name in sleeve], replaced / tranches))
    return out


def relative_metrics(
    dates: list[datetime],
    symbols: list[str],
    probs: np.ndarray,
    next_returns: np.ndarray,
    quantile: float = 0.2,
    cost_bps: float = 5.0,
    tranches: int = 1,
) -> RelativeMetrics:
    """Benchmark-relative and rank-based evaluation of a set of predictions.

    ``tranches`` MUST match the portfolio the gate scores. It did not: the gate
    ran the 10-day tranche book (turnover ~5%) while these metrics ran the
    daily book (turnover ~26%), so a report showed "sharpe 0.79" beside
    "sharpe_net −0.05" for the same window — two different portfolios printed
    as if they described one. IC and the equal-weight benchmark are
    construction-independent; the long/short/net series are not.
    """
    by_date: dict[datetime, list[int]] = defaultdict(list)
    for i, d in enumerate(dates):
        by_date[d].append(i)
    # A one-name cross-section has no ranking and no benchmark to speak of.
    sessions = [d for d in sorted(by_date) if len(by_date[d]) >= 2]

    ics: list[float] = []
    long_returns: list[float] = []
    short_returns: list[float] = []
    bench_returns: list[float] = []
    gross_returns: list[float] = []
    turnovers: list[float] = []
    cost_rate = cost_bps / 10_000.0

    long_book = _tranche_holdings(sessions, by_date, symbols, probs, quantile, tranches)
    short_book = _tranche_holdings(
        sessions, by_date, symbols, probs, quantile, tranches, highest=False
    )

    for idx, session in enumerate(sessions):
        rows = by_date[session]
        p = probs[rows]
        r = next_returns[rows]
        ics.append(_spearman(p, r))

        returns_by_symbol = {symbols[i]: float(next_returns[i]) for i in rows}
        held, turnover = long_book[idx]
        shorted, _ = short_book[idx]
        if not held:
            continue
        gross = float(np.mean([returns_by_symbol.get(name, 0.0) for name in held]))
        gross_returns.append(gross)
        turnovers.append(turnover)
        long_returns.append(gross - cost_rate * turnover)
        short_returns.append(
            float(np.mean([returns_by_symbol.get(name, 0.0) for name in shorted]))
            if shorted
            else 0.0
        )
        bench_returns.append(float(np.mean(r)))  # equal-weight whole universe

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

    daily_returns: list[float] = []
    turnovers: list[float] = []
    position_counts: list[int] = []

    # one definition of "what the book holds" — shared with relative_metrics so
    # the gate and the relative metrics can never drift onto different books
    book = _tranche_holdings(sessions, by_date, symbols, probs, quantile, tranches)
    for idx, session in enumerate(sessions):
        rows = by_date[session]
        returns_by_symbol = {symbols[i]: float(next_returns[i]) for i in rows}
        held, turnover = book[idx]
        if not held:
            continue
        gross = float(np.mean([returns_by_symbol.get(name, 0.0) for name in held]))
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
        returns=tuple(float(r) for r in returns),
    )
