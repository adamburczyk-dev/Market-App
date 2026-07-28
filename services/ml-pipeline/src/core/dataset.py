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

from dataclasses import dataclass, field, replace
from datetime import datetime

import numpy as np
import structlog
from trading_common.features import FEATURE_LOOKBACK, FULL_HISTORY, compute_feature_vector
from trading_common.prices import adjusted_ohlc
from trading_common.ranking import cross_sectional_rank
from trading_common.schemas import OHLCVBar

from src.core.labels import BarrierOutcome, LabelParams, triple_barrier_label
from src.core.uniqueness import average_uniqueness

logger = structlog.get_logger()

# Excluded from the model input:
#  - absolute-level features, whose cross-sectional rank proxies price level, not signal
#  - momentum_20, a deprecated alias of return_20d (identical column — perfect
#    collinearity gives the same information twice and inflates its influence)
EXCLUDED_FEATURES: frozenset[str] = frozenset(
    {"close", "sma_10", "sma_20", "sma_50", "momentum_20"}
)

REGIMES = ("expansion", "recovery", "slowdown", "contraction", "crisis")

# Below this variance a column carries no information and is dropped from the
# model contract (T0-2). Ranks live in [0, 1], so this is far below any real signal.
MIN_FEATURE_VARIANCE = 1e-6


@dataclass(frozen=True)
class DatasetParams:
    label: LabelParams = field(default_factory=LabelParams)
    # P2-1: below FULL_HISTORY a row is missing the slowest features
    # (`momentum_12_1`, `dist_52w_high`) and fills them with the neutral rank
    # 0.5 — a made-up value that is not the median of anything. The floor is the
    # FEATURE requirement, so it moves with the feature set, not with a round
    # number. Both constants come from trading-common so training and serving
    # cannot drift apart (see trading_common.features).
    min_history: int = FULL_HISTORY  # sessions before a symbol contributes rows
    lookback: int = FEATURE_LOOKBACK  # trailing window, exactly as served
    # T0-6: a percentile rank over 2 names is the set {0, 1} — pure noise pooled
    # into the training set. A meaningful quintile needs >= 4 names per bucket.
    min_universe: int = 20  # sessions with a thinner cross-section are skipped


@dataclass
class Dataset:
    x: np.ndarray  # (n_samples, n_features) — ranks / one-hots in [0, 1]
    y: np.ndarray  # (n_samples,) — binary triple-barrier outcome
    next_returns: np.ndarray  # (n_samples,) — 1-session forward return (evaluation only)
    dates: list[datetime]  # per-sample session date (drives purged splits)
    symbols: list[str]
    feature_names: list[str]
    # T0-1: how the labels resolved and what the builder discarded. Without this
    # a degenerate barrier setup (90% vertical) or a thin cross-section is
    # invisible — the dataset just looks smaller than expected.
    label_resolution: dict[str, int] = field(default_factory=dict)
    sessions_skipped_thin: int = 0
    # P0-3: average-uniqueness weight per row. Overlapping labels are not
    # independent evidence; without this the loss counts one market episode
    # `horizon` times over. Empty means "unweighted" (older datasets).
    weights: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    # P0-4: which barrier resolved each row, so the report can show the label
    # mix PER FOLD — a window where 95% of labels time out and one where 55%
    # touch a barrier are different problems that look identical in aggregate.
    barriers: list[str] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return int(self.x.shape[0])


def drop_zero_variance_features(
    dataset: Dataset, min_variance: float = MIN_FEATURE_VARIANCE
) -> tuple[Dataset, list[str]]:
    """Remove constant columns from the model input (T0-2). Returns (dataset, dropped).

    A constant column teaches nothing, but it is worse than useless here: the
    macro one-hots are all-zero in training (no regime history is passed) while
    serving sets one of them to 1.0, so the model would meet an input pattern
    it never saw while learning. Until the regime history exists, these columns
    must LEAVE the feature contract rather than be silently zeroed — the served
    row is assembled from ``feature_names``, so dropping them here keeps
    training and serving on the same schema.
    """
    if dataset.n_samples == 0:
        return dataset, []
    variances = dataset.x.var(axis=0)
    keep = [i for i, v in enumerate(variances) if v > min_variance]
    dropped = [dataset.feature_names[i] for i, v in enumerate(variances) if v <= min_variance]
    if not dropped:
        return dataset, []
    logger.info(
        "Dropped zero-variance features",
        dropped=dropped,
        kept=len(keep),
    )
    # `replace`, not a fresh Dataset: rebuilding it by hand silently reset
    # label_resolution and sessions_skipped_thin to their defaults, so the
    # contract report showed zero barrier resolutions whenever ANY column was
    # dropped — which is every real run, since the macro one-hots are constant.
    # The label diagnostics were blank exactly where they matter most.
    return (
        replace(
            dataset,
            x=dataset.x[:, keep],
            feature_names=[dataset.feature_names[i] for i in keep],
        ),
        dropped,
    )


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
        _, adj_high, adj_low, adj_close = adjusted_ohlc(ordered)
        series[symbol] = {
            # Adjusted OHLC: barriers are compared against highs and lows, so
            # the whole bar must live on one scale (see trading_common.prices).
            "closes": adj_close,
            "highs": adj_high,
            "lows": adj_low,
        }
        index_by_date[symbol] = {b.timestamp: i for i, b in enumerate(ordered)}

    all_dates = sorted({b.timestamp for bars in bars_by_symbol.values() for b in bars})

    resolution = {"upper": 0, "lower": 0, "vertical": 0, "unlabeled": 0}
    sessions_skipped_thin = 0

    rows: list[dict[str, float]] = []
    row_dates: list[datetime] = []
    row_symbols: list[str] = []
    row_labels: list[int] = []
    row_next_returns: list[float] = []
    row_spans: list[tuple[str, int, int]] = []
    row_barriers: list[str] = []

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
            if snapshot:  # a session with symbols, just too few to rank
                sessions_skipped_thin += 1
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
                resolution["unlabeled"] += 1
                continue
            if outcome.touch_index - i >= p.label.horizon:
                resolution["vertical"] += 1
            elif outcome.label == 1:
                resolution["upper"] += 1
            else:
                resolution["lower"] += 1
            closes = series[symbol]["closes"]
            rows.append({**ranked.features, **macro})
            row_dates.append(session)
            row_symbols.append(symbol)
            row_labels.append(outcome.label)
            # (symbol, first, last) index span the label actually occupied —
            # the input to the uniqueness weights below.
            row_spans.append((symbol, i, outcome.touch_index))
            row_barriers.append(
                "vertical"
                if outcome.touch_index - i >= p.label.horizon
                else ("upper" if outcome.label == 1 else "lower")
            )
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
    weights = average_uniqueness(row_spans)

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
        weights=weights,
        barriers=row_barriers,
        label_resolution=resolution,
        sessions_skipped_thin=sessions_skipped_thin,
    )
