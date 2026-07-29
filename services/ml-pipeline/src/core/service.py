"""MLPipelineService — training + drift detection over registered baselines.

Training (plan §6–§7): pull the universe's OHLCV history → build the pooled
cross-sectional dataset (shared feature/rank definitions) → purged
walk-forward → activation-gate report → log the model to MLflow and register
its drift baseline. Promotion to production is a separate, manual call.

Drift: wires the DriftDetector (PSI + KS + rolling-Sharpe/accuracy decay) —
compare current feature/prediction distributions against a model's registered
baseline and publish ModelDriftDetectedEvent when the verdict is actionable.
"""

from dataclasses import asdict
from datetime import datetime
from typing import Any

import numpy as np
import structlog
from trading_common.events import ModelDriftDetectedEvent
from trading_common.schemas import FinancialStatements, Interval

from src.core.capacity import run_capacity_probe
from src.core.data_contract import TrainingDataContract
from src.core.dataset import (
    Dataset,
    DatasetParams,
    build_dataset,
    drop_zero_variance_features,
)
from src.core.evaluation import effective_sample_size
from src.core.fundamentals_client import FundamentalsClient
from src.core.inference_log import InferenceLog
from src.core.market_data_client import MarketDataClient
from src.core.model_store import MlflowModelStore
from src.core.monitoring.drift_detector import DriftDetector, DriftReport
from src.core.outcomes import OutcomeResolver
from src.core.registry import ModelBaseline, ModelRegistry
from src.core.sector_study import run_sector_study
from src.core.serving import ServingEngine
from src.core.target_study import calibrate_barriers, score_targets
from src.core.training import TrainingParams, run_training
from src.core.tuning import SweepReport, run_sweep
from src.events.publisher import Publisher

logger = structlog.get_logger()

BASELINE_SAMPLE_CAP = 500  # reference values kept per feature for drift PSI
NEUTRAL_ACCURACY = 0.5  # used when too few resolved outcomes exist to measure


def _dataset_diagnostics(dataset: Dataset, requested: list[str]) -> dict[str, Any]:
    """What the model was actually trained on — the context a gate report needs.

    A failed gate is only interpretable next to this: too few sessions, a
    lopsided base rate or symbols silently dropped for want of history are
    data problems, not model problems.
    """
    sessions = sorted(set(dataset.dates))
    present = sorted(set(dataset.symbols))
    rows_per_session: dict[datetime, int] = {}
    for session in dataset.dates:
        rows_per_session[session] = rows_per_session.get(session, 0) + 1
    per_session = np.array(list(rows_per_session.values()), dtype=float)
    return {
        "symbols_requested": len(requested),
        "symbols_with_rows": len(present),
        "symbols_missing": sorted(set(requested) - set(present)),
        "sessions": len(sessions),
        "first_session": sessions[0].isoformat() if sessions else None,
        "last_session": sessions[-1].isoformat() if sessions else None,
        "positive_rate": round(float(dataset.y.mean()), 4) if dataset.n_samples else None,
        "rows_per_session_median": float(np.median(per_session)) if len(per_session) else 0.0,
        "n_features": len(dataset.feature_names),
    }


class MLPipelineService:
    def __init__(
        self,
        detector: DriftDetector,
        registry: ModelRegistry,
        publisher: Publisher,
        market_client: MarketDataClient | None = None,
        fundamentals_client: FundamentalsClient | None = None,
        model_store: MlflowModelStore | None = None,
        serving: ServingEngine | None = None,
        inference_log: InferenceLog | None = None,
        resolver: OutcomeResolver | None = None,
        aggregator_client: Any = None,  # AggregatorClient protocol (record_outcome)
        horizon_days: int = 10,
        data_contract: TrainingDataContract | None = None,
        dataset_params: DatasetParams | None = None,
    ) -> None:
        self._detector = detector
        self._registry = registry
        self._publisher = publisher
        self._market = market_client
        self._fundamentals = fundamentals_client
        self._store = model_store
        self._serving = serving
        self._log = inference_log
        self._resolver = resolver
        self._aggregator = aggregator_client
        self._horizon_days = horizon_days
        self._contract = data_contract or TrainingDataContract()
        self._dataset_params = dataset_params or DatasetParams()

    @property
    def registry(self) -> ModelRegistry:
        return self._registry

    @property
    def model_store(self) -> MlflowModelStore | None:
        return self._store

    @property
    def serving(self) -> ServingEngine | None:
        return self._serving

    def promote(self, version: str) -> dict[str, Any]:
        """Manual gate sign-off: alias the version as production + hot-reload serving."""
        if self._store is None:
            raise RuntimeError("model store not configured")
        self._store.promote(version)
        serving_model = self._serving.reload() if self._serving is not None else None
        return {
            "model": self._store.model_name,
            "production_version": version,
            "serving_model_id": serving_model,
        }

    # --- daily monitoring loop (plan ML-3) ---

    async def run_daily_monitor(self) -> dict[str, Any]:
        """Resolve matured outcomes, then drift-check the live serving windows.

        Order matters: resolutions first, so the decay inputs are as fresh as
        possible. With no live data yet (cold start, nothing promoted, empty
        window) the run is a logged no-op. When too few outcomes have resolved
        to measure performance, the decay inputs are NEUTRAL (baseline Sharpe,
        accuracy 0.5) — feature-PSI and prediction-shift checks still run.
        """
        if self._serving is None or not self._serving.active or self._serving.model_id is None:
            logger.info("Daily monitor skipped — serving inactive")
            return {"skipped": "serving_inactive"}
        model_id = self._serving.model_id

        resolved: list[float] = []
        if self._resolver is not None:
            resolved = await self._resolver.resolve_pending(model_id)
            if self._aggregator is not None:
                for signed_return in resolved:
                    await self._aggregator.record_outcome("ml", signed_return)

        if self._log is None:
            return {"model_id": model_id, "outcomes_resolved": len(resolved)}
        window = self._log.feature_window(model_id)
        if not window:
            logger.info("Daily monitor: no served inferences yet", model_id=model_id)
            return {"model_id": model_id, "outcomes_resolved": len(resolved), "skipped": "no_data"}

        baseline = self._registry.get(model_id)
        if baseline is None:
            logger.warning("Daily monitor: no drift baseline registered", model_id=model_id)
            return {
                "model_id": model_id,
                "outcomes_resolved": len(resolved),
                "skipped": "no_baseline",
            }

        metrics_30 = self._log.rolling_metrics(model_id, 30, self._horizon_days)
        metrics_90 = self._log.rolling_metrics(model_id, 90, self._horizon_days)
        if metrics_30 is None:  # not enough outcomes → neutral performance inputs
            sharpe_30, accuracy_30 = baseline.baseline_sharpe, NEUTRAL_ACCURACY
        else:
            sharpe_30, accuracy_30 = metrics_30
        sharpe_90 = metrics_90[0] if metrics_90 is not None else baseline.baseline_sharpe

        report = await self.check_drift(
            model_id,
            current_features=window,
            rolling_sharpe_30d=sharpe_30,
            rolling_sharpe_90d=sharpe_90,
            rolling_accuracy_30d=accuracy_30,
            prediction_current=self._log.prediction_window(model_id),
        )
        return {
            "model_id": model_id,
            "outcomes_resolved": len(resolved),
            "performance_measured": metrics_30 is not None,
            "recommended_action": report.recommended_action if report else None,
            "counts": self._log.counts(model_id),
        }

    # --- training (plan ML-1) ---

    async def build_training_dataset(
        self, symbols: list[str], interval: Interval, limit: int, fundamentals: bool = False
    ) -> tuple[Dataset, int]:
        """Returns (dataset, sessions the bars received should have produced).

        The second value is an INTERNAL consistency expectation, not the request
        size: comparing against `limit` would flag a false violation whenever the
        database legitimately holds less history than asked for. Comparing
        against the bars actually received catches the opposite and more
        insidious failure — the builder losing sessions it was given (e.g.
        timestamps that fail to align across symbols).
        """
        if self._market is None:
            raise RuntimeError("market-data client not configured")
        params = self._dataset_params
        bars_by_symbol = {}
        for symbol in symbols:
            bars = await self._market.get_ohlcv(symbol, interval, limit=limit)
            if bars:
                bars_by_symbol[symbol] = bars
            else:
                logger.warning("No history for symbol — skipped", symbol=symbol)
        longest = max((len(b) for b in bars_by_symbol.values()), default=0)
        # a symbol contributes only past min_history, and the last `horizon`
        # sessions cannot be labeled at all
        expected_sessions = max(0, longest - params.min_history - params.label.horizon)

        panel: dict[str, list[FinancialStatements]] | None = None
        if fundamentals:
            if self._fundamentals is None:
                raise RuntimeError("fundamental-data client not configured")
            panel = await self._fundamentals.panel(sorted(bars_by_symbol))
        return (
            build_dataset(bars_by_symbol, params, fundamentals_by_symbol=panel),
            expected_sessions,
        )

    async def target_study(
        self,
        symbols: list[str],
        interval: Interval,
        limit: int = 1500,
        horizons: tuple[int, ...] = (10, 21, 63),
    ) -> dict[str, Any]:
        """Choose the target: barrier width by label shape, horizon by raw-feature IC.

        Model-free by construction (see core/target_study.py) — nothing is
        fitted, so this consumes none of the gate's n_trials budget.
        """
        if self._market is None:
            raise RuntimeError("market-data client not configured")
        bars_by_symbol = {}
        for symbol in symbols:
            bars = await self._market.get_ohlcv(symbol, interval, limit=limit)
            if bars:
                bars_by_symbol[symbol] = bars
        if not bars_by_symbol:
            raise ValueError("no history for any requested symbol")
        current = self._dataset_params.label
        return {
            "symbols_with_rows": len(bars_by_symbol),
            "current": {
                "horizon": current.horizon,
                "pt_mult": current.pt_mult,
                "excess": current.excess,
            },
            "calibration": calibrate_barriers(bars_by_symbol, horizon=current.horizon),
            "targets": score_targets(
                bars_by_symbol, horizons=horizons, lookback=self._dataset_params.lookback
            ),
        }

    async def sector_study(
        self,
        symbols: list[str],
        interval: Interval,
        sector_by_symbol: dict[str, str | None],
        limit: int = 1500,
    ) -> dict[str, Any]:
        """Does demeaning by sector raise each feature's standalone evidence?

        The E2 gate for P2-2, and model-free like the others: both datasets go
        through the real `build_dataset`, so what is compared is exactly what
        training would consume. Neutralization stays OFF for training until this
        says otherwise — and if it is ever turned on, the serving path has to be
        turned on with it or the two compute different numbers.
        """
        if self._market is None:
            raise RuntimeError("market-data client not configured")
        bars_by_symbol = {}
        for symbol in symbols:
            bars = await self._market.get_ohlcv(symbol, interval, limit=limit)
            if bars:
                bars_by_symbol[symbol] = bars
        if not bars_by_symbol:
            raise ValueError("no history for any requested symbol")
        return run_sector_study(bars_by_symbol, sector_by_symbol, self._dataset_params)

    async def capacity_probe(
        self, symbols: list[str], interval: Interval, limit: int = 1500
    ) -> dict[str, Any]:
        """Answer "underfitting or no signal?" — see core/capacity.py.

        Deliberately does NOT enforce the training data contract: the question
        is about the data that exists, and a dataset too thin to train on can
        still be asked whether anything in it is learnable. Nothing is
        registered or logged — the fitted models are diagnostics, not
        candidates.
        """
        dataset, _ = await self.build_training_dataset(symbols, interval, limit)
        dataset, dropped = drop_zero_variance_features(dataset)
        probe = run_capacity_probe(dataset)
        return {
            "dataset": _dataset_diagnostics(dataset, symbols),
            "dropped_zero_variance": dropped,
            "probe": probe.as_dict(),
        }

    async def tune(
        self,
        symbols: list[str],
        interval: Interval,
        limit: int = 1500,
        n_folds: int = 4,
        params: TrainingParams | None = None,
    ) -> dict[str, Any]:
        """Score several training configurations on the walk-forward folds.

        Selection never sees the holdout, and the report carries `n_trials` so
        the activation gate can deflate for the number of configurations tried.
        Nothing is registered — this chooses a setup, not a model.
        """
        dataset, _ = await self.build_training_dataset(symbols, interval, limit)
        dataset, dropped = drop_zero_variance_features(dataset)
        report: SweepReport = run_sweep(dataset, params, n_folds=n_folds)
        return {
            "dataset": _dataset_diagnostics(dataset, symbols),
            "dropped_zero_variance": dropped,
            "sweep": report.as_dict(),
        }

    async def train(
        self,
        symbols: list[str],
        interval: Interval,
        limit: int = 1500,
        params: TrainingParams | None = None,
        fundamentals: bool = False,
    ) -> dict[str, Any]:
        """Full training pass: dataset → walk-forward gate → registry + baseline.

        The model version is logged to MLflow regardless of the gate outcome
        (a failed gate is a result worth keeping); promotion stays manual.

        ``fundamentals`` joins the point-in-time panel (P2-3). OFF by default:
        per the stage-2 rule a feature family enters only once measured, and the
        per-feature IC table in the report is where that is read.
        """
        dataset, requested_sessions = await self.build_training_dataset(
            symbols, interval, limit, fundamentals=fundamentals
        )
        # T0-2: constant columns leave the feature contract before training, so
        # the model never learns a schema that serving cannot reproduce.
        dataset, dropped_features = drop_zero_variance_features(dataset)
        # T0-1: assert the SHAPE of what arrived before a model is fitted to it.
        # A truncated history or a thin cross-section produces a model that
        # looks trained and means nothing; refuse instead of reporting success.
        contract = self._contract.validate(dataset, requested_sessions=requested_sessions)
        params_used = params or TrainingParams()
        model, report = run_training(dataset, params_used)

        version: str | None = None
        if self._store is not None:
            version = self._store.log_training(model, report)
        else:
            logger.warning("Model store unavailable — training run not persisted")

        model_id = (
            f"{self._store.model_name}@v{version}"
            if self._store is not None and version is not None
            else "unpersisted"
        )
        reference = {
            name: dataset.x[-BASELINE_SAMPLE_CAP:, i].tolist()
            for i, name in enumerate(dataset.feature_names)
        }
        probs = model.predict_proba(dataset.x[-BASELINE_SAMPLE_CAP:])
        self.register_baseline(
            model_id,
            reference,
            baseline_sharpe=report.holdout.portfolio.sharpe,
            prediction_reference=probs.tolist(),
        )
        return {
            "model": self._store.model_name if self._store is not None else "global_v1",
            "version": version,
            "model_id": model_id,
            "samples": dataset.n_samples,
            "features": dataset.feature_names,
            "dataset": _dataset_diagnostics(dataset, symbols),
            "dropped_zero_variance": dropped_features,
            "data_contract": contract,
            "effective_sample_size": asdict(
                effective_sample_size(
                    dataset.dates, dataset.symbols, dataset.next_returns, params_used.horizon
                )
            ),
            "gate": report.as_dict(),
        }

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
