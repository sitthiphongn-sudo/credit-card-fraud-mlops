"""Shared helpers for the monitoring subsystem."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow import MlflowClient
from sklearn.metrics import precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "params.yaml"
BEST_MODEL_PATH = ROOT / "reports" / "experiments" / "best_model.json"
DEFAULT_PERIOD_DIR = ROOT / "monitoring" / "generated"
DEFAULT_REPORT_DIR = ROOT / "reports" / "monitoring"
DEFAULT_TRAIN_PATH = ROOT / "data" / "processed" / "train.csv"
DEFAULT_TEST_PATH = ROOT / "data" / "processed" / "test.csv"

TARGET_COLUMN = "Class"
PREDICTION_COLUMN = "prediction"
PROBABILITY_COLUMN = "prediction_proba"
FEATURE_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]


def load_params(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the project's single source of truth for thresholds and settings."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open(encoding="utf-8") as file:
        params = yaml.safe_load(file)

    if not isinstance(params, dict):
        raise ValueError("configs/params.yaml must contain a YAML mapping at the root")
    return params


def monitoring_params(params: dict[str, Any]) -> dict[str, Any]:
    """Return monitoring thresholds required by the monitoring subsystem."""
    values = params.get("monitoring")
    if not isinstance(values, dict):
        raise ValueError("Missing 'monitoring' section in configs/params.yaml")

    required = {
        "drift_share",
        "max_recall_drop",
        "concept_gap",
        "max_error_rate",
        "stale_after_seconds",
    }
    missing = required - values.keys()
    if missing:
        raise ValueError(f"Missing monitoring parameters: {', '.join(sorted(missing))}")
    return values


def simulation_params(params: dict[str, Any]) -> dict[str, float]:
    """Return and validate parameters used to inject synthetic drift."""
    monitoring = monitoring_params(params)
    values = monitoring.get("simulation")
    if not isinstance(values, dict):
        raise ValueError("Missing 'monitoring.simulation' section in configs/params.yaml")

    required = {
        "amount_multiplier",
        "v14_shift_mean",
        "v17_shift_mean",
        "shift_std",
        "small_amount_quantile",
        "concept_flip_share",
    }
    missing = required - values.keys()
    if missing:
        raise ValueError(f"Missing simulation parameters: {', '.join(sorted(missing))}")

    result: dict[str, float] = {}
    for key in required:
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"monitoring.simulation.{key} must be numeric")
        result[key] = float(value)

    if result["amount_multiplier"] <= 0:
        raise ValueError("amount_multiplier must be greater than 0")
    if result["shift_std"] < 0:
        raise ValueError("shift_std must be greater than or equal to 0")
    if not 0.0 < result["small_amount_quantile"] < 1.0:
        raise ValueError("small_amount_quantile must be between 0 and 1")
    if not 0.0 <= result["concept_flip_share"] <= 1.0:
        raise ValueError("concept_flip_share must be between 0 and 1")

    return result


def load_frame(path: Path) -> pd.DataFrame:
    """Read a CSV and verify model features plus the target are present."""
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    frame = pd.read_csv(path)
    required = set(FEATURE_COLUMNS) | {TARGET_COLUMN}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {', '.join(sorted(missing))}")
    return frame


def load_champion() -> tuple[Any, float, str]:
    """Load the current champion model and its threshold from MLflow Registry."""
    params = load_params()
    registry = params.get("registry")
    if not isinstance(registry, dict):
        raise ValueError("Missing 'registry' section in configs/params.yaml")

    model_name = registry.get("model_name")
    alias = registry.get("champion_alias")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("registry.model_name must be a non-empty string")
    if not isinstance(alias, str) or not alias:
        raise ValueError("registry.champion_alias must be a non-empty string")

    client = MlflowClient()
    model_version = client.get_model_version_by_alias(model_name, alias)
    run = client.get_run(model_version.run_id)

    threshold_value = run.data.params.get("threshold")
    if threshold_value is None:
        raise ValueError(
            f"Champion run {model_version.run_id} does not contain the threshold parameter"
        )

    threshold = float(threshold_value)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Champion threshold must be between 0 and 1, got {threshold}")

    model_uri = f"models:/{model_name}@{alias}"
    model = mlflow.sklearn.load_model(model_uri)
    if not hasattr(model, "predict_proba"):
        raise TypeError("Champion model must provide predict_proba()")

    return model, threshold, str(model_version.version)


def load_model_contract(path: Path = BEST_MODEL_PATH) -> dict[str, Any]:
    """Load the model hand-off file containing baseline metrics."""
    contract = read_json(path)
    if "threshold" not in contract:
        raise ValueError(f"Missing 'threshold' in model hand-off file: {path}")
    return contract


def model_threshold(contract: dict[str, Any]) -> float:
    """Read and validate the threshold stored in the model hand-off file."""
    threshold = float(contract["threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Model threshold must be between 0 and 1, got {threshold}")
    return threshold


def contract_test_recall(contract: dict[str, Any]) -> float | None:
    """Return test recall at the selected threshold when available."""
    metrics = contract.get("metrics")
    if not isinstance(metrics, dict):
        return None

    value = metrics.get("test_recall")
    if value is None:
        return None
    return float(value)


def add_predictions(frame: pd.DataFrame, model: Any, threshold: float) -> pd.DataFrame:
    """Append fraud probability and thresholded prediction to a copy of the frame."""
    missing = set(FEATURE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing model features: {', '.join(sorted(missing))}")

    out = frame.copy()
    probabilities = np.asarray(model.predict_proba(out[FEATURE_COLUMNS]))
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("predict_proba() must return probabilities for both binary classes")

    positive_probability = probabilities[:, 1].astype(float)
    out[PROBABILITY_COLUMN] = positive_probability
    out[PREDICTION_COLUMN] = (positive_probability >= threshold).astype(int)
    return out


def actual_recall(frame: pd.DataFrame) -> float | None:
    """Calculate fraud recall, or None when the window has no positive labels."""
    required = {TARGET_COLUMN, PREDICTION_COLUMN}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing performance columns: {', '.join(sorted(missing))}")

    actual = frame[TARGET_COLUMN].astype(int)
    if not (actual == 1).any():
        return None
    return float(recall_score(actual, frame[PREDICTION_COLUMN].astype(int), zero_division=0))


def actual_precision(frame: pd.DataFrame) -> float | None:
    """Calculate fraud precision, or None when the model predicts no positive rows."""
    required = {TARGET_COLUMN, PREDICTION_COLUMN}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing performance columns: {', '.join(sorted(missing))}")

    predicted = frame[PREDICTION_COLUMN].astype(int)
    if not (predicted == 1).any():
        return None
    actual = frame[TARGET_COLUMN].astype(int)
    return float(precision_score(actual, predicted, zero_division=0))


def prediction_ratios(frame: pd.DataFrame) -> dict[str, float]:
    """Return the share of normal/fraud model predictions in one monitoring period."""
    if PREDICTION_COLUMN not in frame:
        raise ValueError(f"Missing '{PREDICTION_COLUMN}' column")
    if frame.empty:
        return {"normal": 0.0, "fraud": 0.0}

    predictions = frame[PREDICTION_COLUMN].astype(int)
    return {
        "normal": float((predictions == 0).mean()),
        "fraud": float((predictions == 1).mean()),
    }


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object from disk."""
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Write one JSON object with stable, human-readable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")