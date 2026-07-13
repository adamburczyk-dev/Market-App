"""Cross-sectional training dataset: ranked features + triple-barrier labels.

Builds the pooled-universe dataset per docs/ml_integration_plan.md: for every
session date, per-symbol feature vectors are computed from trailing history
with the SHARED definitions (``trading_common.features``) and rank-transformed
across the universe with the SHARED transform (``trading_common.ranking``) —
so training reproduces the serving path bit-for-bit. Each row is labeled with
the triple barrier over the symbol's subsequent path.

Level-type features (absolute price / SMAs) are excluded from the model input:
their cross-sectional rank proxies price level, not signal. Scale-free ratios
stay. A missing feature (e.g. a Tier-2 attribute a symbol doesn't have) fills
with the neutral rank 0.5.
"""

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import structlog
from trading_common.features import compute_feature_vector
from trading_common.ranking import cross_sectional_rank
from trading_common.schemas import OHLCVBar

from src.core.labels import BarrierOutcome, LabelParams, triple_barrier_label

logger = structlog.get_logger()

# Absolute-level features whose cross-sectional rank is a price-level proxy.
EXCLUDED_FEATURES: frozenset[str] = frozenset({"close", "sma_10", "sma_20", "sma_50"})

REGIMES = ("expansion", "recovery", "slowdown", "contraction", "crisis")


@dataclass(frozen=True)
class DatasetParams:
    label: LabelParams = field(default_factory=LabelParams)
    min_history: int = 60  # sessions required before a symbol contributes rows
    lookback: int = 250  # trailing window fed to the feature computation (as served)
    min_universe: int = 2  # sessions with fewer symbols yield no cross-section


@dataclass
class Dataset:
    x: np.ndarray  # (n_samples, n_features) — ranks / one-hots in [0, 1]
    y: np.ndarray  # (n_samples,) — binary triple-barrier outcome
    next_returns: np.ndarray  # (n_samples,) — 1-session forward return (evaluation only)
    dates: list[datetime]  # per-sample session date (drives purged splits)
    symbols: list[str]
    feature_names: list[str]

    @property
    def n_samples(self) -> int:
        return int(self.x.shape[0])


def _regime_one_hot(regime: str | None) -> dict[str, float]:
    return {f"macro_{name}": 1.0 if regime == name else 0.0 for name in REGIMES}


def build_dataset(
    bars_by_symbol: dict[str, list[OHLCVBar]],
    params: DatasetParams | None = None,
    regime_by_date: dict[datetime, str] | None = None,
    feature_names: list[str] | None = None,
) -> Dataset:
    """Assemble the pooled cross-sectional dataset from per-symbol OHLCV history.

    ``regime_by_date`` optionally appends the macro-regime one-hot (an unknown
    or missing regime is all-zeros). ``feature_names`` freezes the column
    order (serving/training contract); by default it is derived as the sorted
    union of ranked feature keys minus ``EXCLUDED_FEATURES``.
    """
    p = params or DatasetParams()

    series: dict[str, dict[str, np.ndarray]] = {}
    index_by_date: dict[str, dict[datetime, int]] = {}
    bars_sorted: dict[str, list[OHLCVBar]] = {}
    for symbol, bars in bars_by_symbol.items():
        ordered = sorted(bars, key=lambda b: b.timestamp)
        bars_sorted[symbol] = ordered
        series[symbol] = {
            "closes": np.array([b.close for b in ordered], dtype=float),
            "highs": np.array([b.high for b in ordered], dtype=float),
            "lows": np.array([b.low for b in ordered], dtype=float),
        }
        index_by_date[symbol] = {b.timestamp: i for i, b in enumerate(ordered)}

    all_dates = sorted({b.timestamp for bars in bars_by_symbol.values() for b in bars})

    rows: list[dict[str, float]] = []
    row_dates: list[datetime] = []
    row_symbols: list[str] = []
    row_labels: list[int] = []
    row_next_returns: list[float] = []

    for session in all_dates:
        # Rank over the FULL feature-bearing cross-section (exactly what serving
        # ranks over); rows that cannot be labeled are dropped only afterwards.
        snapshot = []
        members: list[tuple[str, int]] = []
        for symbol, ordered in bars_sorted.items():
            i = index_by_date[symbol].get(session)
            if i is None or i + 1 < p.min_history:
                continue
            window = ordered[max(0, i - p.lookback + 1) : i + 1]
            snapshot.append(compute_feature_vector(window))
            members.append((symbol, i))

        if len(snapshot) < p.min_universe:
            continue

        macro = _regime_one_hot(regime_by_date.get(session) if regime_by_date else None)
        for ranked, (symbol, i) in zip(cross_sectional_rank(snapshot), members, strict=True):
            outcome: BarrierOutcome | None = triple_barrier_label(
                series[symbol]["closes"],
                series[symbol]["highs"],
                series[symbol]["lows"],
                i,
                p.label,
            )
            if outcome is None:
                continue
            closes = series[symbol]["closes"]
            rows.append({**ranked.features, **macro})
            row_dates.append(session)
            row_symbols.append(symbol)
            row_labels.append(outcome.label)
            # a labeled row always has a next bar (labels need future data);
            # the 1-session forward return feeds the daily-rebalance evaluation
            row_next_returns.append(float(closes[i + 1] / closes[i] - 1.0))

    if feature_names is None:
        keys = {key for row in rows for key in row}
        feature_names = sorted(keys - EXCLUDED_FEATURES)

    x = np.array(
        [[row.get(name, 0.5) for name in feature_names] for row in rows], dtype=float
    ).reshape(len(rows), len(feature_names))
    y = np.array(row_labels, dtype=float)
    next_returns = np.array(row_next_returns, dtype=float)

    logger.info(
        "Dataset built",
        samples=len(rows),
        features=len(feature_names),
        symbols=len(bars_by_symbol),
        sessions=len(all_dates),
        positive_rate=round(float(y.mean()), 4) if len(rows) else None,
    )
    return Dataset(
        x=x,
        y=y,
        next_returns=next_returns,
        dates=row_dates,
        symbols=row_symbols,
        feature_names=feature_names,
    )
