"""MLflow-backed model store (plan §7) — local sqlite backend, alias-based stages.

Every training run is logged (params, gate metrics, artifacts); the artifacts
are the model ``state_dict`` plus a LOAD-BEARING ``metadata.json`` (feature
names/order, temperature, network shape) — loading rebuilds the module from
metadata, and serving must refuse to run when the served feature vector does
not match the artifact's feature list. Promotion uses the registry alias
``production`` (MLflow's stage API is deprecated in favour of aliases).
"""

import json
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import structlog
import torch
from mlflow import MlflowClient

from src.core.model import MlpClassifier, TrainConfig, TrainedModel
from src.core.training import GateReport

logger = structlog.get_logger()

PRODUCTION_ALIAS = "production"


class MlflowModelStore:
    def __init__(
        self,
        tracking_uri: str,
        model_name: str = "global_v1",
        experiment: str = "trading-ml",
    ) -> None:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        self._client = MlflowClient(tracking_uri=tracking_uri)
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def log_training(self, model: TrainedModel, report: GateReport) -> str:
        """Log the run + artifacts, register a new model version; returns it."""
        with mlflow.start_run() as run:
            cfg = model.config
            mlflow.log_params(
                {
                    "hidden": str(cfg.hidden),
                    "dropout": cfg.dropout,
                    "lr": cfg.lr,
                    "weight_decay": cfg.weight_decay,
                    "max_epochs": cfg.max_epochs,
                    "n_features": len(model.feature_names),
                }
            )
            mlflow.log_metrics(
                {
                    "holdout_sharpe": report.holdout.portfolio.sharpe,
                    "holdout_auc": report.holdout.auc,
                    "holdout_brier": report.holdout.brier,
                    "gate_passed": float(report.passed),
                    "temperature": model.temperature,
                    **{f"{f.name}_sharpe": f.portfolio.sharpe for f in report.folds[-3:]},
                }
            )
            with tempfile.TemporaryDirectory() as tmp:
                weights = Path(tmp) / "model.pt"
                torch.save(model.module.state_dict(), weights)
                metadata = Path(tmp) / "metadata.json"
                metadata.write_text(
                    json.dumps(
                        {
                            "feature_names": model.feature_names,
                            "temperature": model.temperature,
                            "hidden": list(cfg.hidden),
                            "dropout": cfg.dropout,
                            "gate": report.as_dict(),
                        },
                        indent=2,
                    )
                )
                mlflow.log_artifact(str(weights), artifact_path="model")
                mlflow.log_artifact(str(metadata), artifact_path="model")

            version = self._client.create_model_version(
                name=self._ensure_registered(),
                source=f"{run.info.artifact_uri}/model",
                run_id=run.info.run_id,
            )
        logger.info(
            "Model version logged",
            model=self._model_name,
            version=version.version,
            gate_passed=report.passed,
        )
        return str(version.version)

    def _ensure_registered(self) -> str:
        try:
            self._client.get_registered_model(self._model_name)
        except mlflow.exceptions.MlflowException:
            self._client.create_registered_model(self._model_name)
        return self._model_name

    def promote(self, version: str) -> None:
        """Point the ``production`` alias at a version (manual gate sign-off)."""
        self._client.set_registered_model_alias(self._model_name, PRODUCTION_ALIAS, version)
        logger.info("Model promoted", model=self._model_name, version=version)

    def production_version(self) -> str | None:
        try:
            mv = self._client.get_model_version_by_alias(self._model_name, PRODUCTION_ALIAS)
        except mlflow.exceptions.MlflowException:
            return None
        return str(mv.version)

    def load(self, version: str) -> tuple[TrainedModel, dict[str, Any]]:
        """Rebuild a TrainedModel from a registered version's artifacts."""
        mv = self._client.get_model_version(self._model_name, version)
        local = mlflow.artifacts.download_artifacts(mv.source)
        metadata = json.loads((Path(local) / "metadata.json").read_text())
        hidden = tuple(metadata["hidden"])
        module = MlpClassifier(len(metadata["feature_names"]), hidden, metadata["dropout"])
        module.load_state_dict(torch.load(Path(local) / "model.pt", weights_only=True))
        module.eval()
        model = TrainedModel(
            module=module,
            temperature=float(metadata["temperature"]),
            feature_names=list(metadata["feature_names"]),
            config=TrainConfig(hidden=hidden, dropout=float(metadata["dropout"])),
        )
        return model, metadata

    def load_production(self) -> tuple[TrainedModel, dict[str, Any]] | None:
        version = self.production_version()
        if version is None:
            return None
        return self.load(version)

    def versions(self) -> list[dict[str, Any]]:
        try:
            found = self._client.search_model_versions(f"name='{self._model_name}'")
        except mlflow.exceptions.MlflowException:
            return []
        production = self.production_version()
        return [
            {
                "version": str(mv.version),
                "run_id": mv.run_id or "",
                "production": str(mv.version) == production,
            }
            for mv in sorted(found, key=lambda m: int(m.version))
        ]
