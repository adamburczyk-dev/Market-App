"""Model drift detection — PSI, KS test, and rolling Sharpe decay."""

from dataclasses import dataclass
from datetime import date

import numpy as np
from scipy import stats


@dataclass
class DriftReport:
    """Result of drift analysis for a single model."""

    model_id: str
    report_date: date
    feature_psi_scores: dict[str, float]
    features_drifted: list[str]
    prediction_distribution_shift: float
    rolling_sharpe_30d: float
    rolling_sharpe_90d: float
    sharpe_decay_pct: float
    rolling_accuracy_30d: float
    needs_retrain: bool
    needs_investigation: bool
    recommended_action: str


class DriftDetector:
    """
    Detects model and feature drift using statistical tests.

    Thresholds:
    - PSI > 0.20 → feature drift (Siddiqi 2006)
    - KS p-value < 0.01 → prediction distribution shift
    - Sharpe decay > 30% → performance degradation
    - Accuracy < 0.48 → worse than random

    Research: Rabanser et al. (2019) "Failing Loudly"
    """

    PSI_THRESHOLD = 0.20
    KS_P_THRESHOLD = 0.01
    SHARPE_DECAY_THRESHOLD = -0.30
    ACCURACY_MIN = 0.48

    def compute_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        bins: int = 10,
    ) -> float:
        """
        Population Stability Index between reference and current distributions.

        PSI = Σ (p_i - q_i) * ln(p_i / q_i)
        where p = current proportions, q = reference proportions.
        """
        eps = 1e-6

        # Use common bin edges from reference distribution
        _, bin_edges = np.histogram(reference, bins=bins)

        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current, bins=bin_edges)

        ref_pct = ref_counts / len(reference) + eps
        cur_pct = cur_counts / len(current) + eps

        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(psi)

    def check_prediction_drift(
        self,
        reference: np.ndarray,
        current: np.ndarray,
    ) -> float:
        """
        KS test p-value for prediction distribution shift.

        Low p-value → distributions differ significantly.
        """
        _, p_value = stats.ks_2samp(reference, current)
        return float(p_value)

    def generate_report(
        self,
        model_id: str,
        feature_psi_scores: dict[str, float],
        prediction_ks_pvalue: float,
        rolling_sharpe_30d: float,
        rolling_sharpe_90d: float,
        baseline_sharpe: float,
        rolling_accuracy_30d: float,
    ) -> DriftReport:
        """Generate a drift report with verdicts."""
        today = date.today()

        # Feature drift
        features_drifted = [
            name for name, psi in feature_psi_scores.items() if psi > self.PSI_THRESHOLD
        ]

        # Sharpe decay
        if baseline_sharpe != 0:
            sharpe_decay_pct = (rolling_sharpe_30d - baseline_sharpe) / abs(baseline_sharpe)
        else:
            sharpe_decay_pct = 0.0

        # Prediction shift magnitude (1 - p_value as shift score)
        prediction_shift = 1.0 - prediction_ks_pvalue

        # Verdicts
        needs_retrain = (
            len(features_drifted) > 0
            or sharpe_decay_pct < self.SHARPE_DECAY_THRESHOLD
            or rolling_accuracy_30d < self.ACCURACY_MIN
        )

        needs_investigation = prediction_ks_pvalue < self.KS_P_THRESHOLD

        # Recommended action
        if needs_retrain:
            recommended_action = "auto_retrain"
        elif needs_investigation:
            recommended_action = "alert_and_monitor"
        else:
            recommended_action = "no_action"

        return DriftReport(
            model_id=model_id,
            report_date=today,
            feature_psi_scores=feature_psi_scores,
            features_drifted=features_drifted,
            prediction_distribution_shift=prediction_shift,
            rolling_sharpe_30d=rolling_sharpe_30d,
            rolling_sharpe_90d=rolling_sharpe_90d,
            sharpe_decay_pct=sharpe_decay_pct,
            rolling_accuracy_30d=rolling_accuracy_30d,
            needs_retrain=needs_retrain,
            needs_investigation=needs_investigation,
            recommended_action=recommended_action,
        )
