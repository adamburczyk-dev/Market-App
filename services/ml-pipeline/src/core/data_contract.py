"""Hard assertions on the training data, run before any model sees it (T0-1).

Two of the worst defects this project has shipped would have been caught by a
single pass over the assembled dataset:

- market-data's cache silently answered a 2000-bar request with 250 bars, so
  training ran on 183 sessions instead of 1438 and reported success;
- the macro one-hots were constant zeros for the whole training set while
  serving set one of them to 1.0.

Neither is visible to a unit test: the code was correct in isolation and the
data was wrong. The contract therefore checks the *shape of what arrived*, not
the behaviour of the code that produced it, and it fails the run rather than
letting an uninterpretable model be trained. Its report is logged to MLflow
even when the run fails — a rejected dataset is a result worth keeping.
"""

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import structlog

from src.core.dataset import Dataset

logger = structlog.get_logger()


class TrainingDataContractError(RuntimeError):
    """The assembled dataset is not fit to train on. Carries the full report."""

    def __init__(self, violations: list[str], report: dict[str, Any]) -> None:
        super().__init__("; ".join(violations))
        self.violations = violations
        self.report = report


@dataclass(frozen=True)
class TrainingDataContract:
    min_sessions: int = 1000
    min_symbols_per_session: int = 20
    max_missing_rate_per_feature: float = 0.10
    min_feature_variance: float = 1e-6
    # A dataset materially smaller than requested means something upstream
    # truncated it — the cache-truncation incident in one number.
    expected_session_tolerance: float = 0.02
    min_samples: int = 1000
    # Sessions dropped for a too-thin cross-section are legitimate — early in a
    # window only the long-lived names have enough history. But if a fifth of
    # the window goes that way the universe is too sparse to rank, which is a
    # real defect and a DIFFERENT one from an upstream truncation. Accounting
    # for skipped sessions must not become a blanket excuse for losing them.
    max_thin_session_rate: float = 0.20

    def validate(
        self,
        dataset: Dataset,
        requested_sessions: int | None = None,
    ) -> dict[str, Any]:
        """Return the contract report, or raise TrainingDataContractError."""
        report = build_report(dataset, requested_sessions)
        violations: list[str] = []

        sessions = report["sessions"]
        if sessions < self.min_sessions:
            violations.append(f"sessions {sessions} < required {self.min_sessions}")

        if dataset.n_samples < self.min_samples:
            violations.append(f"samples {dataset.n_samples} < required {self.min_samples}")

        median_width = report["symbols_per_session_median"]
        if median_width < self.min_symbols_per_session:
            violations.append(
                f"median cross-section {median_width} < required {self.min_symbols_per_session}"
            )

        constant = report["constant_features"]
        if constant:
            violations.append(f"constant features present: {', '.join(constant)}")

        missing = {
            name: rate
            for name, rate in report["neutral_fill_rate"].items()
            if rate > self.max_missing_rate_per_feature
        }
        if missing:
            worst = ", ".join(f"{n} {r:.0%}" for n, r in sorted(missing.items()))
            violations.append(f"features filled with the neutral rank too often: {worst}")

        if requested_sessions:
            # Sessions the BUILDER dropped for having too thin a cross-section
            # are accounted for, not counted as evidence of upstream truncation.
            # They are a known, reported, legitimate reason for the difference —
            # blaming them on the data source refuses a healthy dataset and
            # points the operator at the wrong system. (Seen on the real run:
            # 1450 delivered + 63 skipped-thin = 1513 against 1512 requested,
            # reported as "3.5% short — was the history truncated upstream?".)
            accounted = sessions + dataset.sessions_skipped_thin
            shortfall = 1.0 - accounted / requested_sessions
            if shortfall > self.expected_session_tolerance:
                violations.append(
                    f"got {sessions} sessions (+{dataset.sessions_skipped_thin} skipped as thin), "
                    f"requested ~{requested_sessions} "
                    f"({shortfall:.1%} short — was the history truncated upstream?)"
                )
            thin_rate = dataset.sessions_skipped_thin / requested_sessions
            if thin_rate > self.max_thin_session_rate:
                violations.append(
                    f"{dataset.sessions_skipped_thin} of ~{requested_sessions} sessions "
                    f"({thin_rate:.0%}) had fewer than {self.min_symbols_per_session} symbols — "
                    "the universe is too sparse to rank over this window"
                )

        report["passed"] = not violations
        report["violations"] = violations
        report["contract"] = asdict(self)

        if violations:
            logger.error("Training data contract violated", violations=violations)
            raise TrainingDataContractError(violations, report)
        logger.info(
            "Training data contract passed",
            sessions=sessions,
            samples=dataset.n_samples,
            label_resolution=report["label_resolution_ratio"],
        )
        return report


@dataclass
class _Counter:
    values: dict[str, int] = field(default_factory=dict)


def build_report(dataset: Dataset, requested_sessions: int | None = None) -> dict[str, Any]:
    """Descriptive report — computed even for a dataset that fails the contract."""
    sessions = sorted(set(dataset.dates))
    per_session: dict[Any, int] = {}
    for session in dataset.dates:
        per_session[session] = per_session.get(session, 0) + 1
    widths = np.array(list(per_session.values()), dtype=float) if per_session else np.zeros(1)

    variances = dataset.x.var(axis=0) if dataset.n_samples else np.zeros(len(dataset.feature_names))
    constant = [
        name for name, var in zip(dataset.feature_names, variances, strict=True) if var <= 1e-6
    ]
    # A column left at exactly the neutral rank is an absent attribute, not a
    # measurement — track how often each feature is really just a placeholder.
    neutral_fill = {
        name: float(np.mean(dataset.x[:, i] == 0.5)) if dataset.n_samples else 1.0
        for i, name in enumerate(dataset.feature_names)
    }

    resolved = sum(dataset.label_resolution.get(k, 0) for k in ("upper", "lower", "vertical"))
    ratio = {
        key: (dataset.label_resolution.get(key, 0) / resolved if resolved else 0.0)
        for key in ("upper", "lower", "vertical")
    }

    return {
        "samples": dataset.n_samples,
        "sessions": len(sessions),
        "requested_sessions": requested_sessions,
        "first_session": sessions[0].isoformat() if sessions else None,
        "last_session": sessions[-1].isoformat() if sessions else None,
        "symbols": len(set(dataset.symbols)),
        "symbols_per_session_median": float(np.median(widths)),
        "symbols_per_session_min": float(widths.min()),
        "sessions_skipped_thin": dataset.sessions_skipped_thin,
        "n_features": len(dataset.feature_names),
        "constant_features": constant,
        "neutral_fill_rate": {k: round(v, 4) for k, v in neutral_fill.items()},
        "positive_rate": round(float(dataset.y.mean()), 4) if dataset.n_samples else None,
        "label_resolution": dict(dataset.label_resolution),
        # The headline number for T2-4: barriers that never bind mean the
        # triple-barrier method has degenerated into fixed-horizon labelling.
        "label_resolution_ratio": {k: round(v, 4) for k, v in ratio.items()},
        # P2-3: share of ranked rows that had a filing published by their
        # session. 0.0 means fundamentals were not requested; a low non-zero
        # value means the columns are mostly neutral fill, which is not a
        # feature — and looks exactly like a weak one without this number.
        "fundamental_coverage": dataset.fundamental_coverage,
    }
