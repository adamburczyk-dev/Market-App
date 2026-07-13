"""Tests for the cross-sectional dataset builder (shared features + ranks + labels)."""

from datetime import UTC, datetime, timedelta

import numpy as np
from trading_common.schemas import Interval, OHLCVBar

from src.core.dataset import EXCLUDED_FEATURES, DatasetParams, build_dataset
from src.core.labels import LabelParams

START = datetime(2024, 1, 1, tzinfo=UTC)


def make_bars(symbol: str, closes: list[float]) -> list[OHLCVBar]:
    return [
        OHLCVBar(
            symbol=symbol,
            timestamp=START + timedelta(days=i),
            interval=Interval.D1,
            open=c,
            high=c * 1.005,
            low=c * 0.995,
            close=c,
            volume=1_000_000.0 + i,
            source="test",
        )
        for i, c in enumerate(closes)
    ]


def trending(n: int, drift: float, base: float = 100.0, amp: float = 0.4) -> list[float]:
    return [base * (1 + drift) ** i + (amp if i % 2 else -amp) for i in range(n)]


def universe(n: int = 120) -> dict[str, list[OHLCVBar]]:
    return {
        "UP": make_bars("UP", trending(n, 0.004)),
        "DOWN": make_bars("DOWN", trending(n, -0.004)),
        "FLATISH": make_bars("FLATISH", trending(n, 0.0005)),
    }


PARAMS = DatasetParams(label=LabelParams(), min_history=60, min_universe=2)


def test_dataset_shapes_and_ranges():
    ds = build_dataset(universe(), PARAMS)
    assert ds.n_samples > 0
    assert ds.x.shape == (ds.n_samples, len(ds.feature_names))
    assert len(ds.dates) == len(ds.symbols) == ds.n_samples
    assert set(np.unique(ds.y)) <= {0.0, 1.0}
    assert np.all(ds.x >= 0.0) and np.all(ds.x <= 1.0)  # ranks + one-hots only


def test_level_features_excluded():
    ds = build_dataset(universe(), PARAMS)
    assert not set(ds.feature_names) & EXCLUDED_FEATURES
    assert "momentum_20" in ds.feature_names  # scale-free signal stays


def test_trend_separation_is_learnable():
    """The uptrender's momentum rank should be high where its label is 1 far
    more often than the downtrender's — the dataset must encode the signal."""
    ds = build_dataset(universe(), PARAMS)
    momentum = ds.feature_names.index("momentum_20")
    up_rows = [i for i, s in enumerate(ds.symbols) if s == "UP"]
    down_rows = [i for i, s in enumerate(ds.symbols) if s == "DOWN"]
    assert ds.x[up_rows, momentum].mean() > 0.7
    assert ds.x[down_rows, momentum].mean() < 0.3
    assert ds.y[up_rows].mean() > ds.y[down_rows].mean()


def test_history_edges_are_honest():
    ds = build_dataset(universe(120), PARAMS)
    dates = sorted(set(ds.dates))
    # samples begin once min_history is reached...
    assert dates[0] == START + timedelta(days=59)
    # ...the final session can never be labeled (no future bar at all); later
    # sessions may still appear when a barrier was TOUCHED inside the truncated
    # window — only untouched truncated windows are dropped as unresolved.
    assert dates[-1] < START + timedelta(days=119)


def test_untouched_truncated_tail_is_dropped():
    # calm oscillation → barriers never touched → tail labels need the full
    # 10-session window, so the last `horizon` sessions produce no samples
    calm = {
        "A": make_bars("A", trending(120, 0.0002)),
        "B": make_bars("B", trending(120, -0.0002)),
    }
    ds = build_dataset(calm, PARAMS)
    assert ds.n_samples > 0
    assert sorted(set(ds.dates))[-1] <= START + timedelta(days=120 - 1 - 10)


def test_macro_one_hot_appended():
    regimes = {START + timedelta(days=i): "crisis" for i in range(120)}
    ds = build_dataset(universe(), PARAMS, regime_by_date=regimes)
    crisis = ds.feature_names.index("macro_crisis")
    expansion = ds.feature_names.index("macro_expansion")
    assert np.all(ds.x[:, crisis] == 1.0)
    assert np.all(ds.x[:, expansion] == 0.0)


def test_unknown_regime_is_all_zeros():
    ds = build_dataset(universe(), PARAMS)  # no regime_by_date at all
    macro_cols = [i for i, n in enumerate(ds.feature_names) if n.startswith("macro_")]
    assert macro_cols  # columns exist for schema stability
    assert np.all(ds.x[:, macro_cols] == 0.0)


def test_fixed_feature_names_fill_missing_with_neutral_rank():
    ds = build_dataset(universe(), PARAMS, feature_names=["momentum_20", "f_score"])
    assert ds.feature_names == ["momentum_20", "f_score"]
    f_score = ds.feature_names.index("f_score")
    assert np.all(ds.x[:, f_score] == 0.5)  # attribute absent → neutral rank


def test_single_symbol_universe_yields_nothing():
    ds = build_dataset({"UP": make_bars("UP", trending(120, 0.004))}, PARAMS)
    assert ds.n_samples == 0  # no cross-section → no rows


def test_determinism():
    a = build_dataset(universe(), PARAMS)
    b = build_dataset(universe(), PARAMS)
    assert a.feature_names == b.feature_names
    assert np.array_equal(a.x, b.x)
    assert np.array_equal(a.y, b.y)
