"""Tests for DriftDetector — PSI, KS test, and Sharpe decay monitoring."""

import numpy as np
import pytest

from src.core.monitoring.drift_detector import DriftDetector


class TestComputePSI:
    def setup_method(self):
        self.detector = DriftDetector()
        self.rng = np.random.default_rng(42)

    def test_identical_distributions_near_zero(self):
        data = self.rng.normal(0, 1, 10000)
        psi = self.detector.compute_psi(data, data)
        assert psi == pytest.approx(0.0, abs=0.01)

    def test_shifted_distribution_high_psi(self):
        ref = self.rng.normal(0, 1, 10000)
        cur = self.rng.normal(3, 1, 10000)  # big shift
        psi = self.detector.compute_psi(ref, cur)
        assert psi > 0.20

    def test_moderate_shift(self):
        ref = self.rng.normal(0, 1, 10000)
        cur = self.rng.normal(0.5, 1, 10000)  # small shift
        psi = self.detector.compute_psi(ref, cur)
        assert 0.0 < psi < 1.0

    def test_psi_always_non_negative(self):
        ref = self.rng.normal(0, 1, 1000)
        cur = self.rng.normal(1, 2, 1000)
        psi = self.detector.compute_psi(ref, cur)
        assert psi >= 0


class TestCheckPredictionDrift:
    def setup_method(self):
        self.detector = DriftDetector()
        self.rng = np.random.default_rng(42)

    def test_identical_distributions_high_pvalue(self):
        data = self.rng.normal(0, 1, 1000)
        p = self.detector.check_prediction_drift(data, data)
        assert p > 0.05

    def test_different_distributions_low_pvalue(self):
        ref = self.rng.normal(0, 1, 1000)
        cur = self.rng.normal(5, 1, 1000)
        p = self.detector.check_prediction_drift(ref, cur)
        assert p < 0.01


class TestGenerateReport:
    def setup_method(self):
        self.detector = DriftDetector()

    def test_no_action_report(self):
        report = self.detector.generate_report(
            model_id="model_v1",
            feature_psi_scores={"feat_a": 0.05, "feat_b": 0.10},
            prediction_ks_pvalue=0.50,
            rolling_sharpe_30d=1.2,
            rolling_sharpe_90d=1.1,
            baseline_sharpe=1.0,
            rolling_accuracy_30d=0.60,
        )
        assert report.recommended_action == "no_action"
        assert report.needs_retrain is False
        assert report.needs_investigation is False
        assert len(report.features_drifted) == 0

    def test_alert_and_monitor_on_ks_drift(self):
        """Low KS p-value triggers investigation but not retrain."""
        report = self.detector.generate_report(
            model_id="model_v1",
            feature_psi_scores={"feat_a": 0.05},
            prediction_ks_pvalue=0.005,  # < 0.01
            rolling_sharpe_30d=0.9,
            rolling_sharpe_90d=1.0,
            baseline_sharpe=1.0,
            rolling_accuracy_30d=0.55,
        )
        assert report.needs_investigation is True
        assert report.recommended_action == "alert_and_monitor"

    def test_auto_retrain_on_feature_drift(self):
        report = self.detector.generate_report(
            model_id="model_v1",
            feature_psi_scores={"feat_a": 0.30},  # > 0.20
            prediction_ks_pvalue=0.50,
            rolling_sharpe_30d=1.0,
            rolling_sharpe_90d=1.0,
            baseline_sharpe=1.0,
            rolling_accuracy_30d=0.55,
        )
        assert report.needs_retrain is True
        assert report.recommended_action == "auto_retrain"
        assert "feat_a" in report.features_drifted

    def test_auto_retrain_on_sharpe_decay(self):
        """Sharpe decay > 30% triggers retrain."""
        report = self.detector.generate_report(
            model_id="model_v1",
            feature_psi_scores={},
            prediction_ks_pvalue=0.50,
            rolling_sharpe_30d=0.5,
            rolling_sharpe_90d=0.8,
            baseline_sharpe=1.0,  # decay = (0.5-1.0)/1.0 = -0.50
            rolling_accuracy_30d=0.55,
        )
        assert report.sharpe_decay_pct == pytest.approx(-0.50)
        assert report.needs_retrain is True

    def test_auto_retrain_on_low_accuracy(self):
        report = self.detector.generate_report(
            model_id="model_v1",
            feature_psi_scores={},
            prediction_ks_pvalue=0.50,
            rolling_sharpe_30d=0.9,
            rolling_sharpe_90d=1.0,
            baseline_sharpe=1.0,
            rolling_accuracy_30d=0.45,  # < 0.48
        )
        assert report.needs_retrain is True

    def test_zero_baseline_sharpe_no_division_error(self):
        report = self.detector.generate_report(
            model_id="model_v1",
            feature_psi_scores={},
            prediction_ks_pvalue=0.50,
            rolling_sharpe_30d=0.5,
            rolling_sharpe_90d=0.5,
            baseline_sharpe=0.0,
            rolling_accuracy_30d=0.55,
        )
        assert report.sharpe_decay_pct == 0.0

    def test_negative_baseline_sharpe_worsening(self):
        """Negative baseline getting worse (more negative) triggers retrain."""
        report = self.detector.generate_report(
            model_id="model_v1",
            feature_psi_scores={},
            prediction_ks_pvalue=0.50,
            rolling_sharpe_30d=-0.8,
            rolling_sharpe_90d=-0.6,
            baseline_sharpe=-0.5,
            rolling_accuracy_30d=0.55,
        )
        # decay = (-0.8 - (-0.5)) / 0.5 = -0.6
        assert report.sharpe_decay_pct == pytest.approx(-0.6)
        assert report.needs_retrain is True

    def test_negative_baseline_sharpe_improving(self):
        """Negative baseline improving (less negative) does NOT trigger retrain."""
        report = self.detector.generate_report(
            model_id="model_v1",
            feature_psi_scores={},
            prediction_ks_pvalue=0.50,
            rolling_sharpe_30d=-0.2,
            rolling_sharpe_90d=-0.3,
            baseline_sharpe=-0.5,
            rolling_accuracy_30d=0.55,
        )
        # decay = (-0.2 - (-0.5)) / 0.5 = 0.6 (positive = improvement)
        assert report.sharpe_decay_pct == pytest.approx(0.6)
        assert report.needs_retrain is False

    def test_report_fields_populated(self):
        report = self.detector.generate_report(
            model_id="test_model",
            feature_psi_scores={"a": 0.1},
            prediction_ks_pvalue=0.50,
            rolling_sharpe_30d=1.0,
            rolling_sharpe_90d=0.9,
            baseline_sharpe=1.0,
            rolling_accuracy_30d=0.55,
        )
        assert report.model_id == "test_model"
        assert report.report_date is not None
        assert report.prediction_distribution_shift == pytest.approx(0.50)


class TestThresholdConstants:
    def test_psi_threshold(self):
        assert DriftDetector.PSI_THRESHOLD == 0.20

    def test_ks_p_threshold(self):
        assert DriftDetector.KS_P_THRESHOLD == 0.01

    def test_sharpe_decay_threshold(self):
        assert DriftDetector.SHARPE_DECAY_THRESHOLD == -0.30

    def test_accuracy_min(self):
        assert DriftDetector.ACCURACY_MIN == 0.48
