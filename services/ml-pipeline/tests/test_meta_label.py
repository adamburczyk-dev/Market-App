"""Meta-labeling (P5-1): the filter that decides whether to act on a signal.

The machinery is testable now; whether it HELPS on our data is not, because
that depends on a base model with an edge, which the project does not yet have.
So these tests do two things: pin the construction (only selected rows, labels
net of costs, purged folds), and pin the verdict — because the way this feature
goes wrong is not a crash, it is a report that says "precision improved" about
a book that got worse or vanished.
"""

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from src.core.meta_label import (
    BASE_PROBABILITY_FEATURE,
    MetaParams,
    _verdict,
    build_meta_dataset,
    run_meta_labeling,
)
from src.core.model import TrainConfig

D0 = datetime(2021, 1, 4, tzinfo=UTC)


# --- construction ----------------------------------------------------------


def simple_signals(n_sessions: int = 4, n_symbols: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    dates, symbols, probs, rets, feats = [], [], [], [], []
    for s in range(n_sessions):
        for k in range(n_symbols):
            dates.append(D0 + timedelta(days=s))
            symbols.append(f"S{k}")
            probs.append(float(rng.random()))
            rets.append(float(rng.normal(0, 0.02)))
            feats.append([float(rng.random()), float(rng.random())])
    return dates, symbols, np.array(probs), np.array(rets), np.array(feats)


def test_only_the_rows_the_book_would_hold_become_signals():
    """A filter polices the trades that would actually be taken. Selecting
    globally instead of per session would train it on a different set."""
    dates, symbols, probs, rets, x = simple_signals(n_sessions=4, n_symbols=10)
    meta = build_meta_dataset(x, ["a", "b"], dates, symbols, probs, rets, MetaParams(quantile=0.2))
    assert meta.n_signals == 4 * 2  # ceil(0.2 * 10) per session
    assert len(set(meta.dates)) == 4
    # each session contributed its own top 2, not the global top 8
    for session in set(meta.dates):
        rows = [i for i, d in enumerate(dates) if d == session]
        best = sorted(rows, key=lambda i: -probs[i])[:2]
        picked = [s for s, d in zip(meta.symbols, meta.dates, strict=True) if d == session]
        assert sorted(picked) == sorted(symbols[i] for i in best)


def test_the_base_probability_becomes_a_feature():
    """Without it the filter cannot express "act on strong signals, skip
    marginal ones" — the most obvious thing it might learn."""
    dates, symbols, probs, rets, x = simple_signals()
    meta = build_meta_dataset(x, ["a", "b"], dates, symbols, probs, rets)
    assert meta.feature_names == ["a", "b", BASE_PROBABILITY_FEATURE]
    assert meta.x.shape[1] == 3
    assert float(meta.x[:, -1].max()) <= 1.0


def test_the_label_is_profit_after_costs_not_direction():
    """The one thing the base model never sees. A trade that rose less than it
    cost is a LOSS, and calling it a win is how a filter gets trained to keep
    trades that lose money."""
    dates = [D0, D0]
    symbols = ["CHEAP", "DEAR"]
    x = np.array([[0.5], [0.5]])
    probs = np.array([0.9, 0.9])
    rets = np.array([0.0010, 0.0010])  # +10 bps each

    meta = build_meta_dataset(
        x, ["f"], dates, symbols, probs, rets, MetaParams(quantile=1.0, cost_bps=5.0)
    )
    assert list(meta.y) == [1.0, 1.0]  # 10 bps > 5 bps cost

    # ...and with the P5-2 per-name table, the expensive name flips to a loss
    priced = build_meta_dataset(
        x,
        ["f"],
        dates,
        symbols,
        probs,
        rets,
        MetaParams(quantile=1.0, cost_bps=5.0),
        cost_bps_by_symbol={"CHEAP": 2.0, "DEAR": 30.0},
    )
    by_symbol = dict(zip(priced.symbols, priced.y, strict=True))
    assert by_symbol["CHEAP"] == 1.0
    assert by_symbol["DEAR"] == 0.0


def test_a_name_missing_from_the_cost_table_is_not_free():
    dates, symbols, probs, rets, x = simple_signals(n_sessions=2, n_symbols=5)
    meta = build_meta_dataset(
        x,
        ["a", "b"],
        dates,
        symbols,
        probs,
        rets,
        MetaParams(quantile=1.0, cost_bps=100.0),
        cost_bps_by_symbol={"S0": 1.0},
    )
    default_rows = [i for i, s in enumerate(meta.symbols) if s != "S0"]
    forward = {(d, s): r for d, s, r in zip(dates, symbols, rets, strict=True)}
    for i in default_rows:
        expected = forward[(meta.dates[i], meta.symbols[i])] - 0.01
        assert meta.net_returns[i] == pytest.approx(expected)


def test_no_signals_is_an_empty_dataset_not_a_crash():
    meta = build_meta_dataset(np.zeros((0, 2)), ["a", "b"], [], [], np.zeros(0), np.zeros(0))
    assert meta.n_signals == 0
    assert meta.feature_names[-1] == BASE_PROBABILITY_FEATURE


# --- refusals --------------------------------------------------------------


def test_too_few_signals_refuses_rather_than_fitting():
    dates, symbols, probs, rets, x = simple_signals(n_sessions=5, n_symbols=10)
    meta = build_meta_dataset(x, ["a", "b"], dates, symbols, probs, rets)
    with pytest.raises(ValueError, match="too few"):
        run_meta_labeling(meta, MetaParams(min_signals=200))


def test_a_history_too_short_for_a_walk_forward_refuses():
    dates, symbols, probs, rets, x = simple_signals(n_sessions=60, n_symbols=25)
    meta = build_meta_dataset(x, ["a", "b"], dates, symbols, probs, rets)
    with pytest.raises(ValueError, match="walk-forward"):
        run_meta_labeling(meta, MetaParams(min_signals=10, train_size=504, test_size=63))


def test_a_single_outcome_class_refuses():
    dates = [D0 + timedelta(days=i) for i in range(300)]
    meta = build_meta_dataset(
        np.ones((300, 1)),
        ["f"],
        dates,
        ["S0"] * 300,
        np.ones(300),
        np.full(300, 0.05),  # every trade wins
        MetaParams(quantile=1.0),
    )
    with pytest.raises(ValueError, match="same outcome"):
        run_meta_labeling(meta, MetaParams(min_signals=10))


# --- the verdict, which is the part that can mislead -----------------------


def base_report(**overrides) -> dict:
    report = {
        "kept_share": 0.6,
        "precision_lift": 0.08,
        "meta_auc": 0.62,
        "sharpe_unfiltered": 0.4,
        "sharpe_filtered": 0.9,
        "sharpe_delta": 0.5,
    }
    report.update(overrides)
    return report


def test_a_filter_that_does_not_discriminate_is_called_out():
    verdict = _verdict(base_report(meta_auc=0.48, sharpe_delta=0.9, sharpe_filtered=1.3))
    assert "does not discriminate" in verdict
    assert "Do not enable" in verdict


def test_a_filter_that_keeps_almost_nothing_is_not_a_filter():
    """The failure precision alone cannot see: hit rate way up, book gone."""
    verdict = _verdict(base_report(kept_share=0.04, precision_lift=0.30, sharpe_delta=1.5))
    assert "4% of signals" in verdict
    assert "different strategy" in verdict


def test_precision_without_a_better_book_is_refused():
    """Skipping trades that were going to lose only pays if the kept ones cover
    the ones missed. This is the single most likely way to be fooled here."""
    verdict = _verdict(
        base_report(
            precision_lift=0.12, sharpe_unfiltered=0.8, sharpe_filtered=0.5, sharpe_delta=-0.3
        )
    )
    assert "no better" in verdict
    assert "Do not enable" in verdict


def test_a_genuine_improvement_still_names_the_entry_condition():
    """Even a good result must not read as permission: on a base model with no
    ranking information this exact number is what overfitting produces."""
    verdict = _verdict(base_report())
    assert "Worth enabling ONLY if" in verdict
    assert "G1" in verdict


# --- end to end, where a veto signal genuinely exists ----------------------


def universe_with_a_real_veto(n_sessions: int = 700, n_symbols: int = 20, seed: int = 3):
    """Signals in the "toxic" state lose money; everything else is a coin flip.

    The toxic state is one of the base features, so a filter CAN learn it. This
    is the only condition under which meta-labeling is supposed to help, so it
    is the condition the machinery has to be shown working under.
    """
    rng = np.random.default_rng(seed)
    dates, symbols, probs, rets, feats = [], [], [], [], []
    for s in range(n_sessions):
        for k in range(n_symbols):
            toxic = float(rng.random() < 0.4)
            dates.append(D0 + timedelta(days=s))
            symbols.append(f"S{k}")
            probs.append(float(rng.random()))
            feats.append([toxic, float(rng.random())])
            rets.append(float(rng.normal(-0.02 if toxic else 0.02, 0.01)))
    return dates, symbols, np.array(probs), np.array(rets), np.array(feats)


def test_a_learnable_veto_is_found_and_improves_the_book():
    dates, symbols, probs, rets, x = universe_with_a_real_veto()
    meta = build_meta_dataset(
        x, ["toxic", "noise"], dates, symbols, probs, rets, MetaParams(quantile=0.3)
    )
    report = run_meta_labeling(
        meta,
        MetaParams(
            quantile=0.3,
            train_size=252,
            test_size=63,
            min_signals=100,
            config=TrainConfig(hidden=(8, 4), max_epochs=40, min_epochs=10),
        ),
    )
    assert report["meta_auc"] > 0.6, report
    assert report["precision_lift"] > 0.05, report
    assert report["sharpe_delta"] > 0, report
    assert 0.3 < report["kept_share"] < 0.95, report
    assert report["suggested_n_trials"] >= 1
    assert "Worth enabling ONLY if" in report["verdict"]
