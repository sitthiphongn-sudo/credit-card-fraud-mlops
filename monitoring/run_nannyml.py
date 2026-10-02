"""Estimate recall without labels using NannyML CBPE and compare it with realized recall."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from common import (
    DEFAULT_PERIOD_DIR,
    DEFAULT_REPORT_DIR,
    PREDICTION_COLUMN,
    PROBABILITY_COLUMN,
    TARGET_COLUMN,
    actual_recall,
    add_predictions,
    contract_test_recall,
    load_champion,
    load_frame,
    load_model_contract,
    write_json,
)

PERIOD_FILES = {
    "period_1": "period_1.csv",
    "period_2": "period_2.csv",
    "period_3": "period_3_data_drift.csv",
    "period_4": "period_4_concept_drift.csv",
}


def import_nannyml():
    """Import NannyML lazily because the team has not pinned it in requirements yet."""
    try:
        import nannyml as nml
    except ImportError as exc:
        raise RuntimeError(
            "NannyML is not installed in the approved project environment. "
            "The core drift check can continue without CBPE, but CBPE reports require "
            "a team-approved compatible NannyML dependency."
        ) from exc
    return nml


def nanny_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only columns needed by CBPE and realized performance calculation."""
    return frame[[TARGET_COLUMN, PREDICTION_COLUMN, PROBABILITY_COLUMN]].copy()


def metric_value(frame: pd.DataFrame, metric: str) -> float:
    """Extract the latest metric value from a NannyML result dataframe."""
    if frame.empty:
        return float("nan")

    candidates = [
        (metric, "value"),
        (metric, "realized"),
        f"{metric} value",
        f"{metric}_value",
        "value",
    ]
    for candidate in candidates:
        if candidate in frame.columns:
            series = pd.to_numeric(frame[candidate], errors="coerce").dropna()
            if not series.empty:
                return float(series.iloc[-1])

    for column in frame.columns:
        text = " ".join(map(str, column)) if isinstance(column, tuple) else str(column)
        if metric in text and text.endswith("value"):
            series = pd.to_numeric(frame[column], errors="coerce").dropna()
            if not series.empty:
                return float(series.iloc[-1])

    raise RuntimeError(
        f"Could not find {metric} value in NannyML output columns: {list(frame.columns)}"
    )


def save_plot(figure: Any, path: Path) -> None:
    """Save a Plotly figure as self-contained HTML without adding image dependencies."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(path), include_plotlyjs="cdn")


def run_cbpe(reference: pd.DataFrame, periods: dict[str, pd.DataFrame], report_dir: Path) -> dict:
    """Fit CBPE on healthy labeled rows and compare estimates with realized recall."""
    nml = import_nannyml()
    reference_data = nanny_frame(reference)

    estimator = nml.CBPE(
        y_pred_proba=PROBABILITY_COLUMN,
        y_pred=PREDICTION_COLUMN,
        y_true=TARGET_COLUMN,
        metrics=["recall"],
        problem_type="classification_binary",
        chunk_number=1,
    ).fit(reference_data)

    calculator = nml.PerformanceCalculator(
        y_pred_proba=PROBABILITY_COLUMN,
        y_pred=PREDICTION_COLUMN,
        y_true=TARGET_COLUMN,
        metrics=["recall"],
        problem_type="classification_binary",
        chunk_number=1,
    ).fit(reference_data)

    results: dict[str, Any] = {}
    report_dir.mkdir(parents=True, exist_ok=True)

    for period, frame in periods.items():
        labeled = nanny_frame(frame)
        unlabeled = labeled.drop(columns=[TARGET_COLUMN])

        estimated = estimator.estimate(unlabeled)
        realized = calculator.calculate(labeled)
        estimated_df = estimated.filter(period="analysis").to_df()
        realized_df = realized.filter(period="analysis").to_df()

        estimated_df.to_csv(report_dir / f"{period}_estimated.csv", index=False)
        realized_df.to_csv(report_dir / f"{period}_realized.csv", index=False)
        save_plot(estimated.plot(), report_dir / f"{period}_estimated.html")
        save_plot(realized.plot(), report_dir / f"{period}_realized.html")

        estimated_recall = metric_value(estimated_df, "recall")
        realized_recall = actual_recall(frame)
        gap = estimated_recall - realized_recall if realized_recall is not None else None
        results[period] = {
            "estimated_recall": estimated_recall,
            "actual_recall": realized_recall,
            "concept_gap": float(gap) if gap is not None and np.isfinite(gap) else None,
        }

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NannyML CBPE on simulated periods.")
    parser.add_argument("--period-dir", type=Path, default=DEFAULT_PERIOD_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR / "nannyml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, threshold, model_version = load_champion()
    contract = load_model_contract()

    period_frames = {
        period: add_predictions(load_frame(args.period_dir / filename), model, threshold)
        for period, filename in PERIOD_FILES.items()
    }
    reference = pd.concat(
        [period_frames["period_1"], period_frames["period_2"]],
        ignore_index=True,
    )

    results = run_cbpe(reference, period_frames, args.report_dir)
    output = {
        "metric": "recall",
        "model_threshold": threshold,
        "model_version": model_version,
        "contract_test_recall": contract_test_recall(contract),
        "reference_actual_recall": actual_recall(reference),
        "periods": results,
    }
    write_json(DEFAULT_REPORT_DIR / "nannyml_summary.json", output)
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
