"""The activation gate — six conditions, each answering a different question.

Run #2 is why this exists. It cleared the previous gate ("holdout Sharpe > 0.5
and 2 of the 3 most recent folds") with a holdout AUC of 0.4865 — below a coin
flip — a lift of −0.0003, and an equal-weight universe that out-Sharped it 1.36
to 0.79. An absolute Sharpe on a long-only book in a rising market measures the
market. Each condition here closes one way a model with no edge can look good:

  G0  sanity        — did a model actually get trained and does it vary?
  G1  rank info     — is the ranking distinguishable from noise (IC t-stat)?
  G2  vs baseline   — does it beat the raw rank of a single feature?
  G3  economics     — does the book make money AND beat its own universe?
  G4  calibration   — are the probabilities better than the base rate?
  G5  selection     — does the Sharpe survive deflation for how often we tried?

Conditions are reported individually (with their numbers) rather than collapsed
into a bool, so a failed gate says which question was answered "no".
"""

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.core.evaluation import deflated_sharpe_ratio

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.core.training import FoldReport


@dataclass(frozen=True)
class GateThresholds:
    """Every bar the gate applies. Only three are free parameters."""

    sharpe: float = 0.5  # project rule: no strategy live below OOS Sharpe 0.5
    ic_tstat: float = 2.0  # ~95% that the mean IC is not zero
    # Deflated Sharpe. 0.95 is the textbook bar and it is the right one before
    # live capital; at 0.90 this gate governs promotion to a PAPER-trading vote
    # (one source among several in the aggregator), and the project's separate
    # rule — 30 days of positive paper Sharpe — stands between here and money.
    # With ~600 stitched OOS sessions and 10 trials, 0.90 still demands roughly
    # a 1.8 annualized OOS Sharpe, so this is not a formality.
    dsr: float = 0.90
    # How many model configurations have been tried in total. UNDERSTATING this
    # makes G5 optimistic — it is an honesty input, not a tuning knob. Raise it
    # whenever a new variant is trained against the same data.
    n_trials: int = 10
    min_recent_folds: int = 2
    min_portfolio_sessions: int = 20


@dataclass(frozen=True)
class GateCondition:
    id: str
    name: str
    passed: bool
    detail: str

    def as_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "passed": self.passed, "detail": self.detail}


@dataclass(frozen=True)
class GateOutcome:
    conditions: list[GateCondition] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.conditions)

    @property
    def reasons(self) -> list[str]:
        return [f"{c.id}: {c.detail}" for c in self.conditions if not c.passed]

    def as_dict(self) -> list[dict]:
        return [c.as_dict() for c in self.conditions]


def ic_tstat(ic_mean: float, ic_std: float, n_cross_sections: int) -> float:
    """t-statistic of the mean information coefficient.

    This is the point of measuring IC at all: over 63 sessions the standard
    error of a Sharpe is ~2.0, while the mean IC has an order of magnitude more
    power because every name in every cross-section contributes.
    """
    if ic_std <= 0 or n_cross_sections < 2:
        return 0.0
    return ic_mean / (ic_std / math.sqrt(n_cross_sections))


def evaluate_gate(
    holdout: "FoldReport",
    folds: list["FoldReport"],
    thresholds: GateThresholds | None = None,
) -> GateOutcome:
    """Apply G0–G5 to the holdout (with the folds as the stability check)."""
    t = thresholds or GateThresholds()
    conditions: list[GateCondition] = []
    rel = holdout.relative
    diag = holdout.diagnostics

    # --- G0: is there a model at all? ---
    problems: list[str] = []
    best_epoch = holdout.fit.get("best_epoch")
    if isinstance(best_epoch, int | float) and best_epoch <= 1:
        # Run #2 had two folds like this: 30 epochs ran, validation loss never
        # improved after the first, so the scored weights predate any learning.
        problems.append(f"best epoch {int(best_epoch)} — the fit never improved")
    if diag.pred_std <= 0:
        problems.append("predictions are constant")
    if holdout.portfolio.n_sessions < t.min_portfolio_sessions:
        problems.append(
            f"only {holdout.portfolio.n_sessions} portfolio sessions (< {t.min_portfolio_sessions})"
        )
    if len(folds) < t.min_recent_folds:
        problems.append(f"only {len(folds)} evaluable folds (< {t.min_recent_folds})")
    conditions.append(
        GateCondition(
            "G0",
            "sanity",
            not problems,
            "; ".join(problems) if problems else "model trained, predictions vary, enough windows",
        )
    )

    # --- G1: does the ranking carry information? ---
    if rel is None:
        conditions.append(GateCondition("G1", "rank information", False, "no relative metrics"))
    else:
        tstat = ic_tstat(rel.ic_mean, rel.ic_std, rel.n_cross_sections)
        conditions.append(
            GateCondition(
                "G1",
                "rank information",
                rel.ic_mean > 0 and tstat >= t.ic_tstat,
                f"IC {rel.ic_mean:+.4f}, t={tstat:.2f} (need > 0 and t ≥ {t.ic_tstat})",
            )
        )

    # --- G2: is the ML layer earning its place? ---
    baseline = max(holdout.baseline_ic.values(), default=0.0)
    model_ic = rel.ic_mean if rel is not None else 0.0
    conditions.append(
        GateCondition(
            "G2",
            "beats the single-feature baseline",
            model_ic > baseline,
            f"model IC {model_ic:+.4f} vs best raw feature {baseline:+.4f}",
        )
    )

    # --- G3: does it make money, and is the money its own? ---
    recent = folds[-3:]
    passing = sum(1 for f in recent if f.portfolio.sharpe > t.sharpe)
    active = rel.sharpe_active if rel is not None else 0.0
    benchmark = rel.sharpe_benchmark_ew if rel is not None else 0.0
    economics = (
        holdout.portfolio.sharpe > t.sharpe
        and active > 0.0
        and diag.lift > 0.0
        and passing >= t.min_recent_folds
    )
    conditions.append(
        GateCondition(
            "G3",
            "economics",
            economics,
            f"sharpe {holdout.portfolio.sharpe:.2f} (need > {t.sharpe}), "
            f"active {active:+.2f} vs benchmark {benchmark:.2f}, "
            f"lift {diag.lift:+.4f}, {passing}/{len(recent)} recent folds",
        )
    )

    # --- G4: are the probabilities honest? ---
    # Judged as a PAIRED comparison against predicting the window's base rate
    # every day, with the standard error of that difference. A strict
    # "brier <= base_brier" failed a model with AUC 0.59 on 0.0005 of Brier —
    # noise, not dishonesty. Requiring it not to be SIGNIFICANTLY worse keeps
    # the condition meaningful without inventing a tolerance: a hand-built
    # report with no SE falls back to the strict rule, so it fails closed.
    base_brier = diag.base_rate * (1.0 - diag.base_rate)
    tolerance = 2.0 * holdout.brier_delta_se
    conditions.append(
        GateCondition(
            "G4",
            "calibration",
            holdout.brier_delta <= tolerance and holdout.auc > 0.5,
            f"brier {holdout.brier:.4f} vs base rate {base_brier:.4f} "
            f"(delta {holdout.brier_delta:+.5f} ± {holdout.brier_delta_se:.5f}), "
            f"auc {holdout.auc:.4f}",
        )
    )

    # --- G5: would this Sharpe have appeared anyway, given how often we tried? ---
    # Judged on the STITCHED out-of-sample curve (folds in chronological order,
    # then the holdout) rather than the holdout alone: 126 sessions cannot
    # establish a Sharpe at any useful confidence — the standard error over
    # that window is ~1.4 annualized — and the walk-forward test blocks are
    # disjoint and out-of-sample, so together they are the whole body of
    # evidence the run produced.
    oos: tuple[float, ...] = tuple(r for f in folds for r in f.portfolio.returns) + tuple(
        holdout.portfolio.returns
    )
    dsr = deflated_sharpe_ratio(oos, n_trials=t.n_trials)
    conditions.append(
        GateCondition(
            "G5",
            "deflated sharpe",
            dsr >= t.dsr,
            f"DSR {dsr:.3f} on {len(oos)} stitched OOS sessions over "
            f"{t.n_trials} trials (need ≥ {t.dsr})",
        )
    )

    return GateOutcome(conditions)
