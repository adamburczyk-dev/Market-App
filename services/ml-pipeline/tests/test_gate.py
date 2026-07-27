"""The gate decision, pinned on the numbers of the real run #2.

That run PASSED the Sharpe-only gate while ranking no better than chance and
losing to its own universe. These tests hold the line: the exact numbers from
that report must fail, and each interlock must be the reason.
"""

from src.core.evaluation import PortfolioResult, RelativeMetrics, SelectionDiagnostics
from src.core.training import FoldReport, TrainingParams, gate_reasons

PARAMS = TrainingParams()


def fold(
    name: str,
    *,
    sharpe: float,
    auc: float = 0.60,
    brier: float = 0.24,
    active: float = 1.0,
    benchmark: float = 0.5,
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
            n_sessions=63,
            avg_positions=14.0,
            avg_turnover=0.05,
        ),
        diagnostics=SelectionDiagnostics(
            base_rate=0.55,
            selected_hit_rate=0.58,
            lift=0.03,
            pred_mean=0.5,
            pred_std=0.02,
            pred_p10=0.48,
            pred_p90=0.52,
        ),
        auc_train=0.55,
        relative=RelativeMetrics(
            ic_mean=0.02,
            ic_std=0.25,
            icir=0.08,
            ic_positive_share=0.55,
            n_cross_sections=63,
            sharpe_benchmark_ew=benchmark,
            sharpe_active=active,
            sharpe_long_short=0.5,
            sharpe_gross=sharpe,
            sharpe_net=sharpe,
            cost_drag_annualized=0.03,
            turnover_daily_mean=0.05,
        ),
    )


RUN_2_HOLDOUT = fold(
    "holdout",
    sharpe=0.7923,  # cleared the 0.5 bar
    auc=0.4865,  # ...while ranking worse than a coin flip
    brier=0.2504,
    active=-1.0571,  # ...and losing to its own universe
    benchmark=1.3579,
)
RUN_2_RECENT = [
    fold("fold_5", sharpe=-1.6093),
    fold("fold_6", sharpe=4.5398),
    fold("fold_7", sharpe=2.1879),
]


def test_run_2_numbers_do_not_pass():
    reasons = gate_reasons(RUN_2_HOLDOUT, RUN_2_RECENT, base_rate=0.5518, p=PARAMS)
    assert reasons, "the gate must refuse a model that ranks at chance and trails the benchmark"
    assert any("auc" in r for r in reasons)
    assert any("active sharpe" in r for r in reasons)
    # ...and NOT for the reasons the old gate looked at — those were satisfied.
    assert not any("holdout sharpe" in r for r in reasons)
    assert not any("recent folds" in r for r in reasons)


def test_a_model_with_skill_passes():
    holdout = fold("holdout", sharpe=1.2, auc=0.58, active=0.9, benchmark=0.4)
    recent = [fold(f"fold_{i}", sharpe=1.0) for i in range(3)]
    assert gate_reasons(holdout, recent, base_rate=0.52, p=PARAMS) == []


def test_beating_the_benchmark_is_not_enough_without_discrimination():
    # Active Sharpe positive but AUC at chance: the selection could not have
    # produced the outperformance, so it is not evidence of skill.
    holdout = fold("holdout", sharpe=1.5, auc=0.4990, active=2.0)
    reasons = gate_reasons(holdout, RUN_2_RECENT, base_rate=0.52, p=PARAMS)
    assert reasons == [
        "holdout auc 0.499 ≤ 0.5 — no discrimination on unseen data",
    ]


def test_discrimination_is_not_enough_without_beating_the_benchmark():
    holdout = fold("holdout", sharpe=1.5, auc=0.62, active=-0.2, benchmark=2.0)
    reasons = gate_reasons(holdout, RUN_2_RECENT, base_rate=0.52, p=PARAMS)
    assert len(reasons) == 1
    assert "active sharpe -0.20" in reasons[0]
    assert "benchmark 2.00" in reasons[0]


def test_missing_relative_metrics_fail_closed():
    # No relative metrics → active reads 0.0 → refuse. An unmeasured model is
    # not a passing model.
    holdout = fold("holdout", sharpe=2.0, auc=0.7)
    stripped = FoldReport(
        name=holdout.name,
        n_train=holdout.n_train,
        n_test=holdout.n_test,
        auc=holdout.auc,
        brier=holdout.brier,
        portfolio=holdout.portfolio,
        diagnostics=holdout.diagnostics,
    )
    reasons = gate_reasons(stripped, RUN_2_RECENT, base_rate=0.52, p=PARAMS)
    assert any("active sharpe" in r for r in reasons)
