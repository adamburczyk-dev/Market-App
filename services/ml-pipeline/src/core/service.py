"""MLPipelineService — drift detection over registered model baselines.

Wires the DriftDetector (PSI + KS + rolling-Sharpe/accuracy decay) into the
runtime: compare current feature/prediction distributions against a model's
registered baseline, and publish a ModelDriftDetectedEvent when the verdict is
actionable (retrain or investigate).
"""

import numpy as np
import structlog
from trading_common.events import ModelDriftDetectedEvent

from src.core.monitoring.drift_detector import DriftDetector, DriftReport
from src.core.registry import ModelBaseline, ModelRegistry
from src.events.publisher import Publisher

logger = structlog.get_logger()


class MLPipelineService:
    def __init__(
        self,
        detector: DriftDetector,
        registry: ModelRegistry,
        publisher: Publisher,
    ) -> None:
        self._detector = detector
        self._registry = registry
        self._publisher = publisher

    @property
    def registry(self) -> ModelRegistry:
        return self._registry

    def register_baseline(
        self,
        model_id: str,
        reference_features: dict[str, list[float]],
        baseline_sharpe: float,
        prediction_reference: list[float] | None = None,
    ) -> ModelBaseline:
        baseline = ModelBaseline(
            model_id=model_id,
            reference_features=reference_features,
            baseline_sharpe=baseline_sharpe,
            prediction_reference=prediction_reference or [],
        )
        self._registry.register(baseline)
        logger.info(
            "Registered model baseline",
            model_id=model_id,
            features=sorted(reference_features),
            baseline_sharpe=baseline_sharpe,
        )
        return baseline

    async def check_drift(
        self,
        model_id: str,
        current_features: dict[str, list[float]],
        rolling_sharpe_30d: float,
        rolling_sharpe_90d: float,
        rolling_accuracy_30d: float,
        prediction_current: list[float] | None = None,
    ) -> DriftReport | None:
        """Run the daily drift check for a model; publish an event if actionable.

        Returns the DriftReport, or None if the model has no registered baseline.
        """
        baseline = self._registry.get(model_id)
        if baseline is None:
            logger.warning("Drift check for unknown model", model_id=model_id)
            return None

        # PSI per feature present in BOTH the reference and the current sample.
        feature_psi: dict[str, float] = {}
        for name, ref_values in baseline.reference_features.items():
            cur_values = current_features.get(name)
            if not ref_values or not cur_values:
                continue
            feature_psi[name] = self._detector.compute_psi(
                np.asarray(ref_values, dtype=float),
                np.asarray(cur_values, dtype=float),
            )

        # KS prediction-shift p-value (default: no shift when no prediction samples).
        ks_pvalue = 1.0
        if baseline.prediction_reference and prediction_current:
            ks_pvalue = self._detector.check_prediction_drift(
                np.asarray(baseline.prediction_reference, dtype=float),
                np.asarray(prediction_current, dtype=float),
            )

        report = self._detector.generate_report(
            model_id=model_id,
            feature_psi_scores=feature_psi,
            prediction_ks_pvalue=ks_pvalue,
            rolling_sharpe_30d=rolling_sharpe_30d,
            rolling_sharpe_90d=rolling_sharpe_90d,
            baseline_sharpe=baseline.baseline_sharpe,
            rolling_accuracy_30d=rolling_accuracy_30d,
        )

        if report.recommended_action != "no_action":
            event = self._to_event(report)
            await self._publisher.publish(event)
            logger.info(
                "Model drift detected",
                model_id=model_id,
                drift_type=event.drift_type,
                severity=event.severity,
                action=event.recommended_action,
            )
        else:
            logger.info("No drift detected", model_id=model_id)

        return report

    def _to_event(self, report: DriftReport) -> ModelDriftDetectedEvent:
        """Map a DriftReport's primary trigger to the notification event."""
        if report.features_drifted:
            drift_type = "feature_drift"
        elif report.sharpe_decay_pct < self._detector.SHARPE_DECAY_THRESHOLD:
            drift_type = "performance_decay"
        elif report.rolling_accuracy_30d < self._detector.ACCURACY_MIN:
            drift_type = "accuracy_decay"
        else:
            drift_type = "prediction_shift"

        return ModelDriftDetectedEvent(
            model_id=report.model_id,
            drift_type=drift_type,
            severity="critical" if report.needs_retrain else "warning",
            recommended_action=report.recommended_action,
        )
