"""Generate Evidently drift and classification reports.

Two reference views are retained:

1. Development reference:
   data/processed/train.csv, used to show train-to-production shift.

2. Scenario control reference:
   matched healthy production windows, used to isolate the drift
   intentionally injected into the monitoring demonstration.
"""

from __future__ import annotations

import argparse
import json
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
        from evidently import (
            BinaryClassification,
            DataDefinition,
            Dataset,
            Report,
        )
        from evidently.presets import (
            ClassificationPreset,
            DataDriftPreset,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Evidently is required. "
            "Install the project's pinned requirements."
        ) from exc

    return (
        BinaryClassification,
        DataDefinition,
        Dataset,
        Report,
        ClassificationPreset,
        DataDriftPreset,
    )


def extract_drift_summary(
    payload: dict[str, Any],
    columns: list[str],
) -> dict[str, Any]:
    """Normalize Evidently 0.7.8 drift JSON.

    Evidently 0.7.8 stores metric identity in metric_id,
    while the corresponding human-readable test name is
    available in the tests array. Metrics and tests are
    emitted in matching order.
    """
    metrics = payload.get("metrics", [])
    tests = payload.get("tests", [])

    if not isinstance(metrics, list):
        metrics = []

    if not isinstance(tests, list):
        tests = []

    drifted_count: int | None = None
    drift_share: float | None = None

    feature_scores: dict[str, float] = {}
    drifted_features: list[str] = []

    valid_columns = set(columns)

    summary_prefix = (
        "Share of Drifted Columns"
    )

    feature_prefix = (
        "Value Drift for column "
    )

    for index, test in enumerate(tests):
        if not isinstance(test, dict):
            continue

        name = str(
            test.get("name", "")
        )

        if (
            name.startswith(summary_prefix)
            and index < len(metrics)
        ):
            metric = metrics[index]

            if isinstance(metric, dict):
                value = metric.get("value")

                if isinstance(value, dict):
                    count = value.get("count")
                    share = value.get("share")

                    if (
                        count is not None
                        and share is not None
                    ):
                        drifted_count = int(
                            count
                        )
                        drift_share = float(
                            share
                        )

        if not name.startswith(
            feature_prefix
        ):
            continue

        column = (
            name.removeprefix(
                feature_prefix
            )
            .strip()
            .strip("'\"")
        )

        if column not in valid_columns:
            continue

        status = str(
            test.get("status", "")
        ).upper()

        if status == "FAIL":
            drifted_features.append(
                column
            )

        if index >= len(metrics):
            continue

        metric = metrics[index]

        if not isinstance(metric, dict):
            continue

        value = metric.get("value")

        if isinstance(
            value,
            (int, float),
        ):
            feature_scores[column] = float(
                value
            )

    if (
        drifted_count is None
        or drift_share is None
    ):
        raise RuntimeError(
            "Could not extract drift count/share "
            "from Evidently 0.7.8 JSON output."
        )

    return {
        "drifted_features": sorted(
            set(drifted_features)
        ),
        "drifted_count": drifted_count,
        "drift_share": drift_share,
        "feature_scores": (
            feature_scores
        ),
    }


def classification_dataset(
    frame,
    BinaryClassification,
    DataDefinition,
    Dataset,
):
    """Map project columns to Evidently classification schema."""
    definition = DataDefinition(
        numerical_columns=FEATURE_COLUMNS,
        classification=[
            BinaryClassification(
                target=TARGET_COLUMN,
                prediction_labels=(
                    PREDICTION_COLUMN
                ),
                prediction_probas=(
                    PROBABILITY_COLUMN
                ),
                pos_label=1,
            )
        ],
    )

    return Dataset.from_pandas(
        frame,
        data_definition=definition,
    )


def save_drift_report(
    name: str,
    current,
    reference,
    *,
    report_dir: Path,
    drift_threshold: float,
    suffix: str,
) -> dict[str, Any]:
    """Run one Evidently DataDriftPreset and save HTML/JSON."""
    (
        _,
        _,
        _,
        Report,
        _,
        DataDriftPreset,
    ) = _evidently_imports()

    report = Report(
        [
            DataDriftPreset(
                columns=FEATURE_COLUMNS,
                drift_share=drift_threshold,
            )
        ],
        include_tests=True,
    )

    snapshot = report.run(
        current,
        reference,
    )

    html_path = (
        report_dir
        / f"{name}{suffix}.html"
    )

    json_path = (
        report_dir
        / f"{name}{suffix}.json"
    )

    snapshot.save_html(
        str(html_path)
    )

    snapshot.save_json(
        str(json_path)
    )

    # Read the actual serialized JSON because Evidently 0.7.8
    # produces a slightly different structure from snapshot.dict().
    payload = json.loads(
        json_path.read_text(
            encoding="utf-8"
        )
    )

    return extract_drift_summary(
        payload,
        FEATURE_COLUMNS,
    )


def run_period(
    name: str,
    current,
    control_reference,
    development_reference,
    *,
    report_dir: Path,
    drift_threshold: float,
) -> dict[str, Any]:
    """Run drift and labeled-performance reports for one period."""
    (
        BinaryClassification,
        DataDefinition,
        Dataset,
        Report,
        ClassificationPreset,
        _,
    ) = _evidently_imports()

    # ---------------------------------------------------------
    # 1. Operational / simulation-control drift
    # ---------------------------------------------------------
    control_drift = save_drift_report(
        name,
        current,
        control_reference,
        report_dir=report_dir,
        drift_threshold=drift_threshold,
        suffix="_drift",
    )

    # ---------------------------------------------------------
    # 2. Train-to-production drift
    #
    # This preserves data/processed/train.csv as the
    # model-development reference required by the project.
    # It is evidence only and does not control the simulated
    # OK/WATCH decision.
    # ---------------------------------------------------------
    train_drift = save_drift_report(
        name,
        current,
        development_reference,
        report_dir=report_dir,
        drift_threshold=drift_threshold,
        suffix="_train_reference_drift",
    )

    # ---------------------------------------------------------
    # 3. Classification performance
    # ---------------------------------------------------------
    current_dataset = (
        classification_dataset(
            current,
            BinaryClassification,
            DataDefinition,
            Dataset,
        )
    )

    reference_dataset = (
        classification_dataset(
            control_reference,
            BinaryClassification,
            DataDefinition,
            Dataset,
        )
    )

    # Do not pass include_tests=False here.
    # Evidently 0.7.8 fails inside F1ByLabel in that mode.
    performance_report = Report(
        [
            ClassificationPreset()
        ]
    )

    performance_snapshot = (
        performance_report.run(
            current_dataset,
            reference_dataset,
        )
    )

    performance_html_path = (
        report_dir
        / f"{name}_performance.html"
    )

    performance_json_path = (
        report_dir
        / f"{name}_performance.json"
    )

    performance_snapshot.save_html(
        str(
            performance_html_path
        )
    )

    performance_snapshot.save_json(
        str(
            performance_json_path
        )
    )

    return {
        # Values used by drift_check / decision logic.
        "drifted_features": (
            control_drift[
                "drifted_features"
            ]
        ),
        "drifted_count": (
            control_drift[
                "drifted_count"
            ]
        ),
        "drift_share": (
            control_drift[
                "drift_share"
            ]
        ),
        "feature_scores": (
            control_drift[
                "feature_scores"
            ]
        ),

        # Extra evidence against the original training set.
        "train_reference_drifted_features": (
            train_drift[
                "drifted_features"
            ]
        ),
        "train_reference_drifted_count": (
            train_drift[
                "drifted_count"
            ]
        ),
        "train_reference_drift_share": (
            train_drift[
                "drift_share"
            ]
        ),

        # Model quality.
        "actual_recall": (
            actual_recall(current)
        ),
        "actual_precision": (
            actual_precision(current)
        ),
        "prediction_ratios": (
            prediction_ratios(current)
        ),
        "prediction_probability_mean": float(
            current[
                PROBABILITY_COLUMN
            ].mean()
        ),
    }


def reference_for_period(
    period: str,
    period_dir: Path,
) -> Path:
    """Return the fixed healthy operational reference."""
    return (
        period_dir
        / "period_1.csv"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Evidently monitoring reports."
        )
    )

    parser.add_argument(
        "--period-dir",
        type=Path,
        default=DEFAULT_PERIOD_DIR,
    )

    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_TRAIN_PATH,
        help=(
            "Model-development reference. "
            "Defaults to data/processed/train.csv."
        ),
    )

    parser.add_argument(
        "--report-dir",
        type=Path,
        default=(
            DEFAULT_REPORT_DIR
            / "evidently"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    params = load_params()

    thresholds = monitoring_params(
        params
    )

    drift_threshold = float(
        thresholds["drift_share"]
    )

    (
        model,
        model_threshold,
        model_version,
    ) = load_champion()

    # ---------------------------------------------------------
    # Development reference = training data
    # ---------------------------------------------------------
    development_reference_path = (
        args.reference
    )

    if not development_reference_path.exists():
        raise FileNotFoundError(
            "Development reference not found: "
            f"{development_reference_path}"
        )

    development_reference = (
        add_predictions(
            load_frame(
                development_reference_path
            ),
            model,
            model_threshold,
        )
    )

    args.report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results: dict[str, Any] = {}

    # ---------------------------------------------------------
    # Evaluate every production period
    # ---------------------------------------------------------
    for period, filename in (
        PERIOD_FILES.items()
    ):
        current_path = (
            args.period_dir
            / filename
        )

        control_reference_path = (
            reference_for_period(
                period,
                args.period_dir,
            )
        )

        if not control_reference_path.exists():
            raise FileNotFoundError(
                f"Control reference missing: "
                f"{control_reference_path}. "
                "Run monitoring/simulate_drift.py first."
            )

        current = add_predictions(
            load_frame(current_path),
            model,
            model_threshold,
        )

        control_reference = (
            add_predictions(
                load_frame(
                    control_reference_path
                ),
                model,
                model_threshold,
            )
        )

        print(
            f"Running Evidently for {period} "
            f"(control={control_reference_path.name})..."
        )

        period_result = run_period(
            period,
            current,
            control_reference,
            development_reference,
            report_dir=args.report_dir,
            drift_threshold=(
                drift_threshold
            ),
        )

        period_result[
            "control_reference_path"
        ] = str(
            control_reference_path
        )

        results[period] = (
            period_result
        )

    output = {
        "drift_threshold": (
            drift_threshold
        ),
        "model_threshold": (
            model_threshold
        ),
        "model_version": str(
            model_version
        ),
        "development_reference_path": str(
            development_reference_path
        ),
        "decision_reference_strategy": (
            "fixed healthy period_1 baseline "
            "for operational drift decisions"
        ),
        "periods": results,
    }

    write_json(
        DEFAULT_REPORT_DIR
        / "evidently_summary.json",
        output,
    )

    print(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()