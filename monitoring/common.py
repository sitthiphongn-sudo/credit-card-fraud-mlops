"""Shared helpers for the monitoring subsystem."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "params.yaml"
BEST_MODEL_PATH = ROOT / "reports" / "experiments" / "best_model.json"
DEFAULT_MODEL_PATH = ROOT / "models_cache" / "champion_model" / "model.pkl"
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
    """Return monitoring thresholds already agreed by the team."""
    values = params.get("monitoring")
    if not isinstance(values, dict):
        raise ValueError("Missing 'monitoring' section in configs/params.yaml")

    required = {"drift_share", "max_recall_drop", "concept_gap"}
    missing = required - values.keys()
    if missing:
        raise ValueError(f"Missing monitoring parameters: {', '.join(sorted(missing))}")
    return values


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


def load_model(path: Path = DEFAULT_MODEL_PATH):
    """Load the cached champion sklearn pipeline used by the project."""
    if not path.exists():
        raise FileNotFoundError(
            f"Champion model not found: {path}. Register/cache the champion before monitoring."
        )

    src_path = str(ROOT / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    model = joblib.load(path)
    if not hasattr(model, "predict_proba"):
        raise TypeError("Champion model must provide predict_proba()")
    return model


def load_model_contract(path: Path = BEST_MODEL_PATH) -> dict[str, Any]:
    """Load the team hand-off file containing champion threshold and test metrics."""
    contract = read_json(path)
    if "threshold" not in contract:
        raise ValueError(f"Missing 'threshold' in model hand-off file: {path}")
    return contract


def model_threshold(contract: dict[str, Any]) -> float:
    """Read and validate the classification threshold selected by the model owner."""
    threshold = float(contract["threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Model threshold must be between 0 and 1, got {threshold}")
    return threshold


def contract_test_recall(contract: dict[str, Any]) -> float | None:
    """Return test recall at the selected threshold when the hand-off contains it."""
    metrics = contract.get("metrics")
    if not isinstance(metrics, dict):
        return None

    value = metrics.get("test_recall")
    if value is None:
        return None
    return float(value)


def add_predictions(frame: pd.DataFrame, model, threshold: float) -> pd.DataFrame:
    """Append probability and thresholded prediction without modifying the input frame."""
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
    return float(precision_score(frame[TARGET_COLUMN].astype(int), predicted, zero_division=0))


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
