"""In-memory model-baseline registry.

A lightweight placeholder for the eventual MLflow model registry. Holds, per
model, the *reference* feature distributions and baseline Sharpe captured at
training time — the basis the daily drift check compares current data against.
State is in-memory (re-registered by the training/scheduled job); MLflow-backed
persistence is a follow-up.
"""

from dataclasses import dataclass, field


@dataclass
class ModelBaseline:
    model_id: str
    # reference feature samples captured at training: feature name → list of values
    reference_features: dict[str, list[float]]
    baseline_sharpe: float
    # optional reference prediction distribution (for the KS prediction-shift test)
    prediction_reference: list[float] = field(default_factory=list)


class ModelRegistry:
    def __init__(self) -> None:
        self._baselines: dict[str, ModelBaseline] = {}

    def register(self, baseline: ModelBaseline) -> None:
        self._baselines[baseline.model_id] = baseline

    def get(self, model_id: str) -> ModelBaseline | None:
        return self._baselines.get(model_id)

    def model_ids(self) -> list[str]:
        return sorted(self._baselines)

    def __contains__(self, model_id: object) -> bool:
        return model_id in self._baselines
