"""The G0–G5 gate, pinned from both sides.

From below: the exact numbers of the real run #2 — which PASSED the old
Sharpe-only gate while ranking worse than a coin flip — must fail, and each
condition must fail for its own reason.

From above: the conditions must be jointly satisfiable. A gate nothing can pass
is not a safety property, it is a permanent "no", and it would be indis-
tinguishable from a working gate until the day a real model deserved to pass.
"""

import math
from dataclasses import replace

import numpy as np
import pytest

from src.core.evaluation import (
    PortfolioResult,
    RelativeMetrics,
    SelectionDiagnostics,
    deflated_sharpe_ratio,
    expected_max_sharpe,
)
from src.core.gate import GateThresholds, evaluate_gate, ic_tstat
from src.core.training import FoldReport

THRESHOLDS = GateThresholds()


def series(sharpe: float, n: int = 126, seed: int = 3) -> tuple[float, ...]:
    """A daily return series with (approximately) the requested annual Sharpe."""
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0, 0.01, size=n)
    r = r - r.mean()
    if sharpe != 0.0:
        r = r + sharpe * float(r.std(ddof=1)) / math.sqrt(252)
    return tuple(float(v) for v in r)


def fold(
    name: str,
    *,
    sharpe: float,
    auc: float = 0.60,
    brier: float = 0.24,
    base_rate: float = 0.51,
    lift: float = 0.03,
    ic: float = 0.05,
    ic_std: float = 0.25,
    n_cross_sections: int = 126,
    active: float = 1.0,
    benchmark: float = 0.5,
    baseline_ic: float = 0.001,
    best_epoch: int = 18,
    brier_delta: float | None = None,
    brier_delta_se: float = 0.0,
    pred_std: float = 0.02,
    n_sessions: int = 126,
    seed: int = 3,
) -> FoldReport:
    return FoldReport(
        name=name,
        n_train=700,
        n_test=2000,
        auc=auc,
        brier=brier,
        portfolio=PortfolioResult(
            sharpe=sharpe,
            mean_daily_return=0.0005,
            n_sessions=n_sessions,
            avg_positions=14.0,
            avg_turnover=0.05,
            returns=series(sharpe, n=n_sessions, seed=seed),
        ),
        diagnostics=SelectionDiagnostics(
            base_rate=base_rate,
            selected_hit_rate=base_rate + lift,
            lift=lift,
            pred_mean=0.5,
            pred_std=pred_std,
            pred_p10=0.48,
            pred_p90=0.52,
        ),
        auc_train=0.58,
        # Default: exactly as good as predicting the base rate, so a fixture
        # that does not care about calibration neither passes nor fails on it.
        brier_delta=brier - base_rate * (1.0 - base_rate) if brier_delta is None else brier_delta,
        brier_delta_se=brier_delta_se,
        fit={"best_epoch": best_epoch, "epochs_run": 40},
        relative=RelativeMetrics(
            ic_mean=ic,
            ic_std=ic_std,
            icir=ic / ic_std if ic_std else 0.0,
            ic_positive_share=0.55,
            n_cross_sections=n_cross_sections,
            sharpe_benchmark_ew=benchmark,
            sharpe_active=active,
            sharpe_long_short=0.5,
            sharpe_gross=sharpe,
            sharpe_net=sharpe,
            cost_drag_annualized=0.03,
            turnover_daily_mean=0.05,
        ),
        baseline_ic={"return_20d": baseline_ic},
    )


def failed(outcome) -> set[str]:  # type: ignore[no-untyped-def]
    return {c.id for c in outcome.conditions if not c.passed}


# --- from below: run #2 must not pass -------------------------------------


RUN_2_HOLDOUT = fold(
    "holdout",
    sharpe=0.7923,  # cleared the old 0.5 bar
    auc=0.4865,  # ...while ranking worse than a coin flip
    brier=0.2504,
    base_rate=0.5129,
    lift=-0.0003,
    ic=0.00543,
    ic_std=0.27846,
    n_cross_sections=125,
    active=-1.0571,  # ...and losing to its own universe
    benchmark=1.3579,
    baseline_ic=0.0084,  # ...which the raw feature ranked better than
    pred_std=0.0032,
)
RUN_2_RECENT = [
    fold("fold_5", sharpe=-1.6093),
    fold("fold_6", sharpe=4.5398),
    fold("fold_7", sharpe=2.1879),
]


def test_run_2_fails_on_every_question_it_should():
    outcome = evaluate_gate(RUN_2_HOLDOUT, RUN_2_RECENT, THRESHOLDS)
    assert not outcome.passed
    assert failed(outcome) == {"G1", "G2", "G3", "G4", "G5"}
    # G0 passes: a model WAS trained and its predictions do vary — the failure
    # is not mechanical, which is exactly why the old gate could not see it.
    assert "G0" not in failed(outcome)
    assert any("t=0.2" in r for r in outcome.reasons)  # IC indistinguishable from 0
    assert any("1.36" in r for r in outcome.reasons)  # benchmark named in G3


def test_reasons_name_the_condition():
    outcome = evaluate_gate(RUN_2_HOLDOUT, RUN_2_RECENT, THRESHOLDS)
    assert all(r[:2] in {"G0", "G1", "G2", "G3", "G4", "G5"} for r in outcome.reasons)


# --- from above: the gate must be passable --------------------------------


def test_a_model_with_real_skill_passes_every_condition():
    """The passability test. Numbers a genuinely skilled model would produce:
    IC 0.05 over 126 cross-sections (t ≈ 2.2), a book that beats its universe,
    calibration better than the base rate, and ~2.1 stitched OOS Sharpe over
    630 sessions — enough to survive deflation for 10 trials."""
    holdout = fold(
        "holdout",
        sharpe=2.5,
        auc=0.58,
        brier=0.2450,
        base_rate=0.51,
        lift=0.04,
        ic=0.05,
        ic_std=0.25,
        n_cross_sections=126,
        active=1.1,
        benchmark=0.9,
        baseline_ic=0.01,
    )
    recent = [fold(f"fold_{i}", sharpe=2.0, seed=i) for i in range(4)]
    outcome = evaluate_gate(holdout, recent, THRESHOLDS)
    assert outcome.passed, outcome.reasons


# --- each condition, isolated ---------------------------------------------


def test_g0_catches_a_model_that_never_trained():
    # Run #2 had two folds restored at epoch 1 of 30 — the scored weights
    # predate any learning, and one of them still "earned" Sharpe 2.45.
    holdout = fold("holdout", sharpe=2.2, ic=0.05, best_epoch=1)
    outcome = evaluate_gate(holdout, [fold("f", sharpe=1.0)] * 3, THRESHOLDS)
    assert "G0" in failed(outcome)
    assert any("never improved" in r for r in outcome.reasons)


def test_g1_rejects_an_ic_that_is_not_distinguishable_from_noise():
    # Same IC, fewer cross-sections → the t-statistic, not the level, decides.
    strong = fold("holdout", sharpe=2.2, ic=0.05, n_cross_sections=126)
    weak = fold("holdout", sharpe=2.2, ic=0.05, n_cross_sections=20)
    assert "G1" not in failed(evaluate_gate(strong, [fold("f", sharpe=1.0)] * 3, THRESHOLDS))
    assert "G1" in failed(evaluate_gate(weak, [fold("f", sharpe=1.0)] * 3, THRESHOLDS))


def test_g2_rejects_a_model_that_loses_to_one_raw_feature():
    holdout = fold("holdout", sharpe=2.2, ic=0.05, baseline_ic=0.06)
    outcome = evaluate_gate(holdout, [fold("f", sharpe=1.0)] * 3, THRESHOLDS)
    assert "G2" in failed(outcome)


def test_g2_compares_against_the_best_feature_not_the_first():
    """P2-1 widened the comparator from one declared yardstick to every raw
    feature. A model that beats `return_20d` but loses to `momentum_12_1` used
    to pass this condition; now it cannot, and the reason names the winner."""
    holdout = fold("holdout", sharpe=2.2, ic=0.05)
    beaten = replace(holdout, baseline_ic={"return_20d": 0.01, "momentum_12_1": 0.09})
    outcome = evaluate_gate(beaten, [fold("f", sharpe=1.0)] * 3, THRESHOLDS)
    assert "G2" in failed(outcome)
    assert any("momentum_12_1" in r for r in outcome.reasons)
    # ...and the same model passes once it out-ranks every one of them
    ahead = replace(holdout, baseline_ic={"return_20d": 0.01, "momentum_12_1": 0.03})
    assert "G2" not in failed(evaluate_gate(ahead, [fold("f", sharpe=1.0)] * 3, THRESHOLDS))


def test_g3_rejects_beta_dressed_as_alpha():
    # The run #2 pattern in isolation: healthy absolute Sharpe, negative active.
    holdout = fold("holdout", sharpe=3.0, ic=0.05, active=-0.4, benchmark=4.0)
    outcome = evaluate_gate(holdout, [fold("f", sharpe=1.0)] * 3, THRESHOLDS)
    assert "G3" in failed(outcome)


def test_g4_uses_the_windows_own_base_rate():
    # The old gate compared the holdout's Brier against the base rate of the
    # WHOLE dataset (0.5518 → 0.2473) and allowed a further 0.01 of slack, so
    # nothing realistic could fail it. Judged against the holdout's own base
    # rate (0.5129 → 0.2498), a Brier of 0.2499 is worse than predicting the
    # base rate every day — which is what it is.
    holdout = fold("holdout", sharpe=2.2, ic=0.05, brier=0.2499, base_rate=0.5129)
    outcome = evaluate_gate(holdout, [fold("f", sharpe=1.0)] * 3, THRESHOLDS)
    assert "G4" in failed(outcome)


def test_g5_deflates_a_sharpe_found_by_trying_many_times():
    holdout = fold("holdout", sharpe=2.0, ic=0.05, n_sessions=126)
    folds = [fold(f"f{i}", sharpe=2.0, seed=i) for i in range(4)]
    few = evaluate_gate(holdout, folds, GateThresholds(n_trials=2))
    many = evaluate_gate(holdout, folds, GateThresholds(n_trials=500))
    assert "G5" not in failed(few)
    assert "G5" in failed(many), "trying 500 configurations must raise the bar"


# --- the statistics themselves --------------------------------------------


def test_ic_tstat_scales_with_the_number_of_cross_sections():
    assert ic_tstat(0.05, 0.25, 126) == pytest.approx(0.05 / (0.25 / math.sqrt(126)))
    assert ic_tstat(0.05, 0.0, 126) == 0.0  # degenerate → no evidence
    assert ic_tstat(0.05, 0.25, 1) == 0.0


def test_expected_max_sharpe_grows_with_trials():
    assert expected_max_sharpe(1) == 0.0  # nothing was selected
    assert expected_max_sharpe(10) < expected_max_sharpe(1000)


def test_deflated_sharpe_needs_more_than_a_high_ratio():
    strong = series(3.0, n=630, seed=1)
    weak = series(0.8, n=630, seed=1)
    assert deflated_sharpe_ratio(strong, n_trials=10) > 0.95
    assert deflated_sharpe_ratio(weak, n_trials=10) < 0.95
    # the same ratio on a shorter sample is weaker evidence, not equal evidence
    assert deflated_sharpe_ratio(series(3.0, n=126, seed=1), n_trials=10) < deflated_sharpe_ratio(
        strong, n_trials=10
    )
    # too short to judge, and a flat series, both refuse rather than guess
    assert deflated_sharpe_ratio(series(3.0, n=5), n_trials=10) == 0.0
    assert deflated_sharpe_ratio((0.01,) * 30, n_trials=10) == 0.0


# --- P3-4: the decision metric is the relative book ------------------------


def test_g3_leads_with_the_active_sharpe_not_the_absolute_one():
    """A book that makes money by being long in a rising market, and picks worse
    than the universe it picks from, must fail on the HEADLINE number.

    Breadth in a long-only book saturates at ~1/rho, so adding names does not
    add bets — the absolute Sharpe is mostly the market's. Before P3-4 that
    number led the condition and the relative one was a footnote; run #2 cleared
    0.79 absolute while losing to its universe by 1.06.
    """
    holdout = fold("holdout", sharpe=2.5, ic=0.05, active=-0.3, benchmark=3.0)
    outcome = evaluate_gate(holdout, [fold("f", sharpe=2.0)] * 3, THRESHOLDS)
    assert "G3" in failed(outcome)
    reason = next(r for r in outcome.reasons if r.startswith("G3"))
    assert reason.index("active") < reason.index("absolute"), "active must be reported first"


def test_g3_still_requires_the_absolute_floor():
    """Beating the universe is necessary, not sufficient: the project rule
    "no strategy live below OOS Sharpe 0.5" applies to what is actually traded,
    and a book that loses money slightly less than the market still loses."""
    holdout = fold("holdout", sharpe=-0.8, ic=0.05, active=0.6, benchmark=-1.5)
    assert "G3" in failed(evaluate_gate(holdout, [fold("f", sharpe=2.0)] * 3, THRESHOLDS))


def test_g3_recent_folds_are_judged_on_active_too():
    """The stability check moves with the decision metric. Folds with a healthy
    absolute Sharpe that all lose to their universe are not evidence of skill."""
    holdout = fold("holdout", sharpe=2.5, ic=0.05, active=1.1, benchmark=0.9)
    beta_folds = [fold(f"f{i}", sharpe=3.0, active=-0.5, benchmark=4.0, seed=i) for i in range(3)]
    outcome = evaluate_gate(holdout, beta_folds, THRESHOLDS)
    assert "G3" in failed(outcome)
    assert any("0/3 recent folds active-positive" in r for r in outcome.reasons)
