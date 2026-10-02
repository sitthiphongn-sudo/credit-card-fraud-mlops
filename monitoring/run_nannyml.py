"""Run NannyML CBPE using precomputed champion predictions.

This script is designed to run inside the isolated .venv-nannyml
environment. It never loads the champion model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TARGET_COLUMN = "Class"
PREDICTION_COLUMN = "prediction"
PROBABILITY_COLUMN = "prediction_proba"

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_DIR = (
    ROOT
    / "monitoring"
    / "generated"
    / "nannyml_inputs"
)

DEFAULT_REPORT_DIR = (
    ROOT
    / "reports"
    / "monitoring"
)

PERIOD_FILES = {
    "period_1": "period_1.csv",
    "period_2": "period_2.csv",
    "period_3": "period_3.csv",
    "period_4": "period_4.csv",
}


def import_nannyml():
    """Import NannyML."""
    try:
        import nannyml as nml
    except ImportError as exc:
        raise RuntimeError(
            "NannyML is unavailable. "
            "Run this script with .venv-nannyml."
        ) from exc

    return nml


def read_json(
    path: Path,
) -> dict[str, Any]:
    """Read a JSON object."""
    if not path.exists():
        raise FileNotFoundError(
            f"JSON file not found: {path}"
        )

    value = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(value, dict):
        raise ValueError(
            f"Expected JSON object: {path}"
        )

    return value


def write_json(
    path: Path,
    value: dict[str, Any],
) -> None:
    """Write one JSON object."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def load_nanny_frame(
    path: Path,
) -> pd.DataFrame:
    """Load prepared prediction data."""
    if not path.exists():
        raise FileNotFoundError(
            f"NannyML input not found: {path}"
        )

    frame = pd.read_csv(
        path
    )

    required = {
        TARGET_COLUMN,
        PREDICTION_COLUMN,
        PROBABILITY_COLUMN,
    }

    missing = (
        required
        - set(frame.columns)
    )

    if missing:
        raise ValueError(
            f"Missing columns in {path}: "
            + ", ".join(sorted(missing))
        )

    frame[TARGET_COLUMN] = (
        frame[TARGET_COLUMN]
        .astype(int)
    )

    frame[PREDICTION_COLUMN] = (
        frame[PREDICTION_COLUMN]
        .astype(int)
    )

    frame[PROBABILITY_COLUMN] = (
        frame[PROBABILITY_COLUMN]
        .astype(float)
    )

    return frame


def actual_recall(
    frame: pd.DataFrame,
) -> float | None:
    """Calculate realized recall once labels are available."""
    actual_positive = (
        frame[TARGET_COLUMN] == 1
    )

    positive_count = int(
        actual_positive.sum()
    )

    if positive_count == 0:
        return None

    true_positive = int(
        (
            actual_positive
            & (
                frame[
                    PREDICTION_COLUMN
                ]
                == 1
            )
        ).sum()
    )

    return (
        true_positive
        / positive_count
    )


def metric_value(
    frame: pd.DataFrame,
    metric: str,
) -> float:
    """Extract metric value from a NannyML result dataframe."""
    if frame.empty:
        return float("nan")

    candidates: list[
        str | tuple[str, str]
    ] = [
        (metric, "value"),
        (metric, "realized"),
        f"{metric} value",
        f"{metric}_value",
        "value",
    ]

    for candidate in candidates:
        if candidate not in frame.columns:
            continue

        series = pd.to_numeric(
            frame[candidate],
            errors="coerce",
        ).dropna()

        if not series.empty:
            return float(
                series.iloc[-1]
            )

    for column in frame.columns:
        if isinstance(
            column,
            tuple,
        ):
            text = " ".join(
                map(str, column)
            )
        else:
            text = str(column)

        normalized = (
            text.lower()
        )

        if (
            metric.lower()
            not in normalized
        ):
            continue

        if "value" not in normalized:
            continue

        series = pd.to_numeric(
            frame[column],
            errors="coerce",
        ).dropna()

        if not series.empty:
            return float(
                series.iloc[-1]
            )

    raise RuntimeError(
        f"Could not find {metric} value "
        "in NannyML output columns: "
        f"{list(frame.columns)}"
    )


def analysis_dataframe(
    result,
) -> pd.DataFrame:
    """Return analysis rows from a NannyML result."""
    try:
        filtered = result.filter(
            period="analysis"
        )

        frame = filtered.to_df()

        if not frame.empty:
            return frame
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
    ):
        pass

    return result.to_df()


def save_plot(
    result,
    path: Path,
) -> None:
    """Save NannyML plot to HTML."""
    try:
        figure = result.plot()

        figure.write_html(
            str(path),
            include_plotlyjs="cdn",
        )
    except Exception as exc:
        print(
            "WARNING: could not save "
            f"{path.name}: {exc}"
        )


def run_cbpe(
    reference: pd.DataFrame,
    periods: dict[str, pd.DataFrame],
    report_dir: Path,
) -> dict[str, Any]:
    """Fit CBPE and estimate recall without analysis labels."""
    nml = import_nannyml()

    estimator = nml.CBPE(
        y_pred_proba=(
            PROBABILITY_COLUMN
        ),
        y_pred=PREDICTION_COLUMN,
        y_true=TARGET_COLUMN,
        metrics=["recall"],
        problem_type=(
            "classification_binary"
        ),
        chunk_number=1,
    )

    estimator.fit(
        reference
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results: dict[
        str,
        Any,
    ] = {}

    for period, labeled in (
        periods.items()
    ):
        print(
            f"Running NannyML CBPE "
            f"for {period}..."
        )

        # This is the production view:
        # CBPE receives no ground-truth target.
        unlabeled = labeled.drop(
            columns=[TARGET_COLUMN]
        )

        estimated = estimator.estimate(
            unlabeled
        )

        estimated_df = (
            analysis_dataframe(
                estimated
            )
        )

        estimated_csv = (
            report_dir
            / f"{period}_estimated.csv"
        )

        estimated_html = (
            report_dir
            / f"{period}_estimated.html"
        )

        estimated_df.to_csv(
            estimated_csv,
            index=False,
        )

        save_plot(
            estimated,
            estimated_html,
        )

        estimated_recall = (
            metric_value(
                estimated_df,
                "recall",
            )
        )

        realized_recall = (
            actual_recall(
                labeled
            )
        )

        if (
            realized_recall is None
            or not np.isfinite(
                estimated_recall
            )
        ):
            concept_gap = None
        else:
            concept_gap = float(
                estimated_recall
                - realized_recall
            )

        results[period] = {
            "estimated_recall": float(
                estimated_recall
            ),
            "actual_recall": (
                realized_recall
            ),
            "concept_gap": (
                concept_gap
            ),
        }

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run isolated NannyML CBPE."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
    )

    parser.add_argument(
        "--report-dir",
        type=Path,
        default=(
            DEFAULT_REPORT_DIR
            / "nannyml"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    metadata = read_json(
        args.input_dir
        / "metadata.json"
    )

    reference = load_nanny_frame(
        args.input_dir
        / "reference.csv"
    )

    periods = {
        period: load_nanny_frame(
            args.input_dir
            / filename
        )
        for period, filename
        in PERIOD_FILES.items()
    }

    results = run_cbpe(
        reference,
        periods,
        args.report_dir,
    )

    output = {
        "metric": "recall",
        "model_threshold": (
            metadata.get(
                "model_threshold"
            )
        ),
        "model_version": (
            metadata.get(
                "model_version"
            )
        ),
        "contract_test_recall": (
            metadata.get(
                "contract_test_recall"
            )
        ),
        "reference_actual_recall": (
            actual_recall(
                reference
            )
        ),
        "analysis_labels_hidden_from_cbpe": True,
        "periods": results,
    }

    output_path = (
        DEFAULT_REPORT_DIR
        / "nannyml_summary.json"
    )

    write_json(
        output_path,
        output,
    )

    print()
    print(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        )
    )

    print()
    print(
        "NannyML summary written to "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()