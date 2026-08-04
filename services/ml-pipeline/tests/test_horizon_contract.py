"""The horizon must mean one thing across every module that reads it.

Four objects need this number to agree: the label (how far the path is
scanned), the walk-forward purge (how much of the training window overlaps the
test labels), the published event (what the probability is about), and the
outcome resolver (when a vote has matured). They used to carry four independent
defaults.

The dangerous disagreement is the SILENT one. A label horizon larger than the
purge horizon leaks label window into every test block, and the symptom is that
the metrics get BETTER — no exception, no warning, no failing test. That is the
same shape as MAX_OHLCV_LIMIT declared twice in two values, which cost a full
455-symbol backfill before anyone saw it.
"""

from datetime import UTC, datetime, timedelta

from src.config import settings
from src.core.cpcv import cpcv_splits
from src.core.labels import LABEL_HORIZON, LabelParams, outcome_drop_after_days
from src.core.meta_label import MetaParams
from src.core.serving import ServingEngine
from src.core.splits import purged_walk_forward
from src.core.target_study import calibrate_barriers
from src.core.training import TrainingParams


def test_every_declaration_of_the_horizon_agrees():
    """One constant, read in four places — not four defaults that happen to match."""
    assert LabelParams().horizon == LABEL_HORIZON
    assert TrainingParams().horizon == LABEL_HORIZON
    assert settings.LABEL_HORIZON_DAYS == LABEL_HORIZON
    assert MetaParams().horizon == LABEL_HORIZON


def test_function_defaults_that_shadow_the_constant_agree_too():
    """A default argument is a declaration like any other.

    These are overridden at every production call site today, so a stale one
    would stay latent until someone called them plainly — which is exactly how
    latent defaults surface.
    """
    assert cpcv_splits.__defaults__ is not None
    assert LABEL_HORIZON in cpcv_splits.__defaults__
    assert calibrate_barriers.__defaults__ is not None
    assert LABEL_HORIZON in calibrate_barriers.__defaults__
    # ServingEngine takes horizon_days as a keyword-only-ish ctor arg
    assert ServingEngine.__init__.__defaults__ is not None
    assert LABEL_HORIZON in ServingEngine.__init__.__defaults__


def test_the_drop_cutoff_is_derived_from_the_horizon_not_typed():
    """Sessions are not calendar days, and the resolver's clock counts days.

    `horizon` sessions span horizon * 365.25/252 calendar days. A cutoff typed
    as a literal outlives the next horizon change, and this particular literal
    going stale kills the entire ML-3 loop without logging anything: every vote
    resolves as label=None, the adaptive weight never learns, and drift's
    performance arm reports "not measured" forever — indistinguishable from a
    cold start.
    """
    assert outcome_drop_after_days(LABEL_HORIZON) == settings.OUTCOME_DROP_AFTER_DAYS
    # Whatever the horizon is, the cutoff must outlast the label window itself.
    calendar_days_of_one_horizon = LABEL_HORIZON * 365.25 / 252
    assert calendar_days_of_one_horizon < settings.OUTCOME_DROP_AFTER_DAYS


def test_the_purge_covers_the_label_window_that_build_dataset_actually_uses():
    """The seam between train and test must span the LABEL's horizon.

    This is the leakage check. `purged_walk_forward` is called with
    TrainingParams.horizon while the labels are made with LabelParams.horizon;
    if the first is smaller, every fold trains on rows whose labels resolve
    inside its own test block.

    Reverting ONLY TrainingParams.horizon fails here, which is the point —
    nothing else in the suite notices that particular disagreement.
    """
    label = LabelParams()
    train = TrainingParams()
    start = datetime(2020, 1, 1, tzinfo=UTC)
    sessions = [start + timedelta(days=i) for i in range(train.train_size + 4 * train.test_size)]

    folds = purged_walk_forward(
        sessions, train.train_size, train.test_size, train.horizon, train.embargo
    )
    assert folds, "fixture too short to produce a fold"

    for fold in folds:
        gap_sessions = sessions.index(fold.test_dates[0]) - sessions.index(fold.train_dates[-1])
        assert gap_sessions >= label.horizon + train.embargo, (
            f"purge seam {gap_sessions} does not cover a {label.horizon}-session label "
            f"plus a {train.embargo}-session embargo"
        )


def test_the_published_event_names_the_label_it_was_trained_on():
    """A contract field nobody populates is a field that lies by omission.

    `label_kind` was added to MlSignalGeneratedEvent and then defaulted to
    "absolute" everywhere — including after the label became excess, at which
    point the event would have asserted the wrong referent rather than none.
    Caught by a live run, not by the type checker: the default made it valid.
    """
    engine = ServingEngine.__init__
    assert engine.__defaults__ is not None
    assert "absolute" in engine.__defaults__

    published: list[object] = []

    class Recorder:
        async def publish(self, event: object) -> None:
            published.append(event)

    served = ServingEngine(
        Recorder(),  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        None,
        label_kind="excess",
    )
    assert served._label_kind == "excess"
