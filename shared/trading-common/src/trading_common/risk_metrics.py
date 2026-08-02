"""Portfolio risk statistics computed from a realized equity path — SHARED.

These live in trading-common for the same reason `features` does: the dashboard
renders them and risk-mgmt reasons about them, and two implementations of "what
drawdown means" would eventually disagree about whether a limit was breached.

Everything here is **historical**, not parametric: VaR is an empirical quantile
of realized returns, not σ·z. A normal assumption on daily equity returns
understates exactly the tail the number exists to describe, and we have the
actual path — there is no reason to model what we can count.

Every function states its own sample floor. A VaR from 12 observations is not a
conservative VaR, it is a number with no sampling distribution, so these return
``None`` rather than something a chart would happily draw.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

# Below this many returns an empirical quantile is not a measurement. 20 daily
# observations put the 95% VaR on the single worst point — the estimate moves
# entirely with one outlier.
MIN_VAR_SAMPLES = 20

# A correlation from a handful of overlapping days is dominated by whatever the
# market did that week, not by the pair's relationship.
MIN_CORRELATION_SAMPLES = 20


def returns_from_equity(equity: Sequence[float]) -> list[float]:
    """Simple period-over-period returns of an equity path.

    Non-positive equity ends the series: a return through zero is not a return,
    and carrying it forward would produce a ±inf that silently poisons every
    statistic downstream.
    """
    out: list[float] = []
    for previous, current in zip(equity, equity[1:], strict=False):
        if previous <= 0:
            break
        out.append(current / previous - 1.0)
    return out


def drawdown_series(equity: Sequence[float]) -> list[float]:
    """Fractional drawdown from the running peak, one value per point (>= 0)."""
    out: list[float] = []
    peak = -math.inf
    for value in equity:
        peak = max(peak, value)
        out.append((peak - value) / peak if peak > 0 else 0.0)
    return out


def max_drawdown(equity: Sequence[float]) -> float:
    series = drawdown_series(equity)
    return max(series) if series else 0.0


def _tail_count(n: int, confidence: float) -> int:
    """How many observations fall in the loss tail at `confidence`.

    The rounding is not cosmetic: `(1.0 - 0.95) * 100` is 5.000000000000004 in
    binary floating point, so a bare `ceil` returns 6 and the quantile lands one
    past the tail — on the canonical 95% case, of all places. That produced a
    VaR of 0.0 on a series that lost money on five days.
    """
    return max(math.ceil(round((1.0 - confidence) * n, 9)), 1)


def historical_var(returns: Sequence[float], confidence: float = 0.95) -> float | None:
    """Empirical VaR as a POSITIVE loss fraction, or None on too few samples.

    Returned positive because "VaR is 2.1%" means a 2.1% loss; a signed quantile
    invites a chart to plot the tail above the axis.
    """
    if len(returns) < MIN_VAR_SAMPLES or not 0.0 < confidence < 1.0:
        return None
    ordered = sorted(returns)
    # The worst return inside the best `confidence` fraction — i.e.
    # (1 - confidence) of days are at least this bad. The `floor` convention
    # would instead land on the 6th-worst of 100 and report a VaR the five
    # actual bad days already exceeded.
    index = _tail_count(len(ordered), confidence) - 1
    return max(-ordered[index], 0.0)


def conditional_var(returns: Sequence[float], confidence: float = 0.95) -> float | None:
    """Mean loss GIVEN the VaR threshold was breached (expected shortfall).

    VaR says how bad a bad day is at the threshold; it says nothing about what
    lies beyond it, which is the part that ends a portfolio.
    """
    if len(returns) < MIN_VAR_SAMPLES or not 0.0 < confidence < 1.0:
        return None
    ordered = sorted(returns)
    # The same tail VaR's index sits at the edge of, averaged.
    tail = ordered[: _tail_count(len(ordered), confidence)]
    return max(-sum(tail) / len(tail), 0.0)


def annualized_sharpe(returns: Sequence[float], periods_per_year: int = 252) -> float | None:
    """Sharpe of a realized path, or None when the sample cannot support one."""
    if len(returns) < 2:
        return None
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    sd = math.sqrt(variance)
    if sd == 0:
        return None
    return mean / sd * math.sqrt(periods_per_year)


def correlation(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Pearson correlation over the OVERLAPPING prefix, or None if too short."""
    n = min(len(a), len(b))
    if n < MIN_CORRELATION_SAMPLES:
        return None
    x, y = list(a[-n:]), list(b[-n:])
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y, strict=True))
    sxx = sum((xi - mx) ** 2 for xi in x)
    syy = sum((yi - my) ** 2 for yi in y)
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / math.sqrt(sxx * syy)


@dataclass(frozen=True)
class CorrelationMatrix:
    """Pairwise correlations plus how much of the grid could actually be filled.

    `coverage` exists because a matrix of mostly-None is visually
    indistinguishable from a matrix of mostly-zero once it is rendered as a
    heatmap, and those mean opposite things.
    """

    symbols: list[str]
    matrix: list[list[float | None]]
    samples: int
    coverage: float

    def as_dict(self) -> dict[str, object]:
        return {
            "symbols": self.symbols,
            "matrix": self.matrix,
            "samples": self.samples,
            "coverage": self.coverage,
        }


def correlation_matrix(returns_by_symbol: Mapping[str, Sequence[float]]) -> CorrelationMatrix:
    """Correlation grid over the symbols given, in a stable (sorted) order."""
    symbols = sorted(returns_by_symbol)
    grid: list[list[float | None]] = []
    filled = off_diagonal = 0
    for row_symbol in symbols:
        row: list[float | None] = []
        for col_symbol in symbols:
            if row_symbol == col_symbol:
                row.append(1.0)
                continue
            value = correlation(returns_by_symbol[row_symbol], returns_by_symbol[col_symbol])
            off_diagonal += 1
            if value is not None:
                filled += 1
            row.append(value)
        grid.append(row)
    samples = min((len(v) for v in returns_by_symbol.values()), default=0)
    coverage = filled / off_diagonal if off_diagonal else 1.0
    return CorrelationMatrix(symbols=symbols, matrix=grid, samples=samples, coverage=coverage)


def average_pairwise_correlation(matrix: CorrelationMatrix) -> float | None:
    """Mean off-diagonal correlation — the concentration number that matters.

    A book of ten names with ρ = 0.9 is one bet with a wide spreadsheet; the
    effective number of bets is roughly 1/ρ (see `docs/decisions/06`).
    """
    values: list[float] = []
    for i in range(len(matrix.symbols)):
        for j in range(len(matrix.symbols)):
            value = matrix.matrix[i][j]
            if i != j and value is not None:
                values.append(value)
    if not values:
        return None
    return sum(values) / len(values)
