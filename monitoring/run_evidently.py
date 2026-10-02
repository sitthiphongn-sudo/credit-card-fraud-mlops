"""Generate Evidently drift and classification reports for every production period."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from common import (
    DEFAULT_PERIOD_DIR,
    DEFAULT_REPORT_DIR,
    DEFAULT_TRAIN_PATH,
    FEATURE_COLUMNS,
    PREDICTION_COLUMN,
    PROBABILITY_COLUMN,
    TARGET_COLUMN,
    actual_precision,
    actual_recall,
    add_predictions,
    load_champion,
    load_frame,
    load_params,
    monitoring_params,
    prediction_ratios,
    write_json,
)

PERIOD_FILES = {
    "period_1": "period_1.csv",
    "period_2": "period_2.csv",
    "period_3": "period_3_data_drift.csv",
    "period_4": "period_4_concept_drift.csv",
}


def _evidently_imports():
    """Import the pinned Evidently package lazily."""
    try:
        from evidently import BinaryClassification, DataDefinition, Dataset, Report
        from evidently.presets import ClassificationPreset, DataDriftPreset
    except ImportError as exc:
        raise RuntimeError(
            "Evidently is required. Install requirements.txt (evidently==0.7.8)."
        ) from exc

    return (
        BinaryClassification,
        DataDefinition,
        Dataset,
        Report,
        ClassificationPreset,
        DataDriftPreset,
    )


def _find_drift_table(value: Any) -> dict[str, Any] | None:
    """Find the legacy drift_by_columns shape if Evidently includes it."""
    if isinstance(value, dict):
        table = value.get("drift_by_columns")
        if isinstance(table, dict):
            return table
        for child in value.values():
            found = _find_drift_table(child)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_drift_table(child)
            if found is not None:
                return found
    return None


def _metric_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = payload.get("metrics", [])
    if not isinstance(metrics, list):
        return []
    return [metric for metric in metrics if isinstance(metric, dict)]


def _extract_column_from_metric_name(name: str) -> str | None:
    """Extract a column name from common ValueDrift metric-name formats."""
    match = re.search(r"column=['\"]?([^,'\")]+)", name)
    return match.group(1) if match else None


def _failed_drift_features(payload: dict[str, Any], columns: list[str]) -> list[str]:
    """Read failed per-column ValueDrift tests from Evidently 0.7.x output."""
    tests = payload.get("tests", [])
    if not isinstance(tests, list):
        return []

    drifted: list[str] = []
    valid_columns = set(columns)
    prefix = "Value Drift for column "

    for test in tests:
        if not isinstance(test, dict):
            continue
        if str(test.get("status", "")).upper() != "FAIL":
            continue

        name = str(test.get("name", ""))
        if name.startswith(prefix):
            column = name.removeprefix(prefix).strip().strip("'\"")
            if column in valid_columns:
                drifted.append(column)

    return sorted(set(drifted))


def extract_drift_summary(payload: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    """Normalize Evidently output into a small stable JSON contract."""
    drift_table = _find_drift_table(payload)
    if drift_table is not None:
        drifted = sorted(
            name
            for name, item in drift_table.items()
            if isinstance(item, dict) and bool(item.get("drift_detected"))
        )
        scores = {
            name: item.get("drift_score")
            for name, item in drift_table.items()
            if isinstance(item, dict) and item.get("drift_score") is not None
        }
        return {
            "drifted_features": drifted,
            "drifted_count": len(drifted),
            "drift_share": len(drifted) / len(columns) if columns else 0.0,
            "feature_scores": scores,
        }

    drifted_count = None
    drift_share = None
    feature_scores: dict[str, float] = {}

    for metric in _metric_items(payload):
        name = str(metric.get("metric_name", ""))
        value = metric.get("value")
        if name.startswith("DriftedColumnsCount") and isinstance(value, dict):
            drifted_count = value.get("count")
            drift_share = value.get("share")
        elif name.startswith("ValueDrift") and isinstance(value, (int, float)):
            column = _extract_column_from_metric_name(name)
            if column:
                feature_scores[column] = float(value)

    if drifted_count is None or drift_share is None:
        raise RuntimeError(
            "Could not extract drift count/share from Evidently output. "
            "Inspect the generated JSON for the pinned Evidently version."
        )

    drifted_features = _failed_drift_features(payload, columns)
    return {
        "drifted_features": drifted_features,
        "drifted_count": int(drifted_count),
        "drift_share": float(drift_share),
        "feature_scores": feature_scores,
    }


def classification_dataset(frame, BinaryClassification, DataDefinition, Dataset):
    """Map project columns to Evidently's binary-classification schema."""
    definition = DataDefinition(
        numerical_columns=FEATURE_COLUMNS,
        classification=[
            BinaryClassification(
                target=TARGET_COLUMN,
                prediction_labels=PREDICTION_COLUMN,
                prediction_probas=PROBABILITY_COLUMN,
                pos_label=1,
            )
        ],
    )
    return Dataset.from_pandas(frame, data_definition=definition)


def run_period(
    name: str,
    current,
    reference,
    *,
    report_dir: Path,
    drift_threshold: float,
) -> dict[str, Any]:
    """Run drift and labeled-performance reports for one production period."""
    (
        BinaryClassification,
        DataDefinition,
        Dataset,
        Report,
        ClassificationPreset,
        DataDriftPreset,
    ) = _evidently_imports()

    drift_report = Report(
        [DataDriftPreset(columns=FEATURE_COLUMNS, drift_share=drift_threshold)],
        include_tests=True,
    )
    drift_snapshot = drift_report.run(current, reference)
    drift_snapshot.save_html(str(report_dir / f"{name}_drift.html"))
    drift_snapshot.save_json(str(report_dir / f"{name}_drift.json"))
    drift_summary = extract_drift_summary(drift_snapshot.dict(), FEATURE_COLUMNS)

    current_dataset = classification_dataset(
        current,
        BinaryClassification,
        DataDefinition,
        Dataset,
    )
    reference_dataset = classification_dataset(
        reference,
        BinaryClassification,
        DataDefinition,
        Dataset,
    )
    performance_report = Report([ClassificationPreset(include_tests=False)])
    performance_snapshot = performance_report.run(current_dataset, reference_dataset)
    performance_snapshot.save_html(str(report_dir / f"{name}_performance.html"))
    performance_snapshot.save_json(str(report_dir / f"{name}_performance.json"))

    return {
        **drift_summary,
        "actual_recall": actual_recall(current),
        "actual_precision": actual_precision(current),
        "prediction_ratios": prediction_ratios(current),
        "prediction_probability_mean": float(current[PROBABILITY_COLUMN].mean()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Evidently monitoring reports.")
    parser.add_argument("--period-dir", type=Path, default=DEFAULT_PERIOD_DIR)
    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_TRAIN_PATH,
        help="Reference CSV from model development; defaults to data/processed/train.csv.",
    )
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR / "evidently")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = load_params()
    drift_threshold = float(monitoring_params(params)["drift_share"])
    model, model_threshold, model_version = load_champion()

    reference_path = args.reference
    if not reference_path.exists():
        demo_reference = args.period_dir / "reference.csv"
        if demo_reference.exists():
            print(
                f"WARNING: {reference_path} is missing; using {demo_reference} "
                "for a local smoke test only."
            )
            reference_path = demo_reference
        else:
            raise FileNotFoundError(f"Reference data not found: {reference_path}")

    reference = add_predictions(load_frame(reference_path), model, model_threshold)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}
    for period, filename in PERIOD_FILES.items():
        current = add_predictions(
            load_frame(args.period_dir / filename),
            model,
            model_threshold,
        )
        print(f"Running Evidently for {period}...")
        results[period] = run_period(
            period,
            current,
            reference,
            report_dir=args.report_dir,
            drift_threshold=drift_threshold,
        )

    output = {
        "drift_threshold": drift_threshold,
        "model_threshold": model_threshold,
        "model_version": model_version,
        "reference_path": str(reference_path),
        "periods": results,
    }
    write_json(DEFAULT_REPORT_DIR / "evidently_summary.json", output)
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
