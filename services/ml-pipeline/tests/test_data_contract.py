"""T0-1: the training data contract — assertions on the data, not the code.

Every case here corresponds to a defect this project actually shipped or came
close to shipping, and none of them would fail a unit test of the code that
produced the data.
"""

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from src.core.data_contract import TrainingDataContract, TrainingDataContractError, build_report
from src.core.dataset import Dataset

START = datetime(2024, 1, 1, tzinfo=UTC)


def make_dataset(
    sessions: int = 1200,
    width: int = 30,
    n_features: int = 6,
    constant_column: bool = False,
    neutral_column: bool = False,
    seed: int = 3,
) -> Dataset:
    """A synthetic dataset of the right SHAPE — content is irrelevant here."""
    rng = np.random.default_rng(seed)
    dates: list[datetime] = []
    symbols: list[str] = []
    for s in range(sessions):
        for k in range(width):
            dates.append(START + timedelta(days=s))
            symbols.append(f"S{k}")
    n = len(dates)
    x = rng.random((n, n_features))
    names = [f"feat_{i}" for i in range(n_features)]
    if constant_column:
        x = np.column_stack([x, np.zeros(n)])
        names.append("macro_crisis")
    if neutral_column:
        x = np.column_stack([x, np.full(n, 0.5)])
        names.append("f_score")
    return Dataset(
        x=x,
        y=(rng.random(n) > 0.45).astype(float),
        next_returns=rng.normal(0, 0.01, n),
        dates=dates,
        symbols=symbols,
        feature_names=names,
        label_resolution={"upper": 300, "lower": 250, "vertical": 5000, "unlabeled": 120},
    )


def test_healthy_dataset_passes():
    report = TrainingDataContract().validate(make_dataset())
    assert report["passed"] is True
    assert report["violations"] == []
    assert report["sessions"] == 1200
    assert report["symbols_per_session_median"] == 30


def test_contract_rejects_constant_feature():
    """The macro one-hots were constant zeros through a whole training run."""
    with pytest.raises(TrainingDataContractError) as exc:
        TrainingDataContract().validate(make_dataset(constant_column=True))
    assert any("constant features" in v for v in exc.value.violations)
    assert "macro_crisis" in exc.value.report["constant_features"]
    assert exc.value.report["passed"] is False  # report survives the failure


def test_contract_rejects_truncated_history():
    """market-data's cache answered a 2000-bar request with 250 bars; training
    ran on 183 sessions and reported success."""
    with pytest.raises(TrainingDataContractError) as exc:
        TrainingDataContract().validate(make_dataset(sessions=183), requested_sessions=1930)
    joined = " ".join(exc.value.violations)
    assert "sessions 183" in joined
    assert "short" in joined  # the requested-vs-received check fires too


def test_contract_rejects_thin_cross_section():
    with pytest.raises(TrainingDataContractError) as exc:
        TrainingDataContract().validate(make_dataset(width=4))
    assert any("cross-section" in v for v in exc.value.violations)


def test_contract_rejects_feature_that_is_mostly_neutral_fill():
    """A column left at the neutral rank is an absent attribute, not a
    measurement — a Tier-2 feature with poor coverage must not pass silently."""
    with pytest.raises(TrainingDataContractError) as exc:
        TrainingDataContract().validate(make_dataset(neutral_column=True))
    assert any("neutral rank" in v for v in exc.value.violations)
    assert exc.value.report["neutral_fill_rate"]["f_score"] == 1.0


def test_report_carries_label_resolution_ratio():
    """The headline number for T2-4: barriers that never bind mean the triple
    barrier has degenerated into fixed-horizon labelling."""
    report = build_report(make_dataset())
    ratio = report["label_resolution_ratio"]
    assert ratio["vertical"] == pytest.approx(5000 / 5550, abs=1e-3)
    assert ratio["upper"] + ratio["lower"] + ratio["vertical"] == pytest.approx(1.0, abs=1e-6)
    assert report["label_resolution"]["unlabeled"] == 120


def test_report_is_computed_for_a_failing_dataset():
    """A rejected dataset still produces a full report — it is logged to MLflow
    so a refused run leaves the same evidence a successful one does."""
    report = build_report(make_dataset(sessions=10, width=2))
    assert report["sessions"] == 10
    assert report["symbols_per_session_median"] == 2
    assert report["n_features"] == 6


# --- sessions the builder itself dropped (found on the first real 20y run) ---


def thin_dataset(sessions: int, skipped_thin: int) -> Dataset:
    from dataclasses import replace as dc_replace

    return dc_replace(make_dataset(sessions=sessions), sessions_skipped_thin=skipped_thin)


def test_sessions_skipped_as_thin_are_accounted_for_not_blamed_upstream():
    """The real run's numbers. 1450 sessions were delivered, the builder had
    itself dropped 63 for having under 20 symbols, and 1512 were requested:
    1450 + 63 = 1513, so nothing was truncated at all. The contract refused the
    run anyway with "3.5% short — was the history truncated upstream?", which
    both blocked a healthy dataset and pointed at the wrong system.
    """
    report = TrainingDataContract().validate(
        thin_dataset(sessions=1450, skipped_thin=63), requested_sessions=1512
    )
    assert report["passed"] is True, report["violations"]


def test_a_genuinely_truncated_history_is_still_caught():
    """The check must keep doing its job — this is the cache-truncation
    incident, where 1505 stored bars turned into 183 sessions."""
    with pytest.raises(TrainingDataContractError):
        TrainingDataContract().validate(
            thin_dataset(sessions=1100, skipped_thin=0), requested_sessions=1512
        )


def test_accounting_for_thin_sessions_is_not_a_blanket_excuse():
    """If a fifth of the window is being dropped, the universe is too sparse to
    rank over it. That is a real defect — just a different one from upstream
    truncation, and it must not hide behind the accounting."""
    with pytest.raises(TrainingDataContractError) as excinfo:
        TrainingDataContract().validate(
            thin_dataset(sessions=1100, skipped_thin=412), requested_sessions=1512
        )
    assert "too sparse to rank" in str(excinfo.value)
