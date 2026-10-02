"""Prepare prediction data for NannyML in an isolated environment.

This script must run in the project's main virtual environment because
it loads the registered champion model.

NannyML itself does not run here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from common import (
    DEFAULT_PERIOD_DIR,
    PREDICTION_COLUMN,
    PROBABILITY_COLUMN,
    ROOT,
    TARGET_COLUMN,
    add_predictions,
    load_champion,
    load_frame,
)

PERIOD_FILES = {
    "period_1": "period_1.csv",
    "period_2": "period_2.csv",
    "period_3": "period_3_data_drift.csv",
    "period_4": "period_4_concept_drift.csv",
}

DEFAULT_OUTPUT_DIR = (
    ROOT
    / "monitoring"
    / "generated"
    / "nannyml_inputs"
)

BEST_MODEL_PATH = (
    ROOT
    / "reports"
    / "experiments"
    / "best_model.json"
)


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object."""
    if not path.exists():
        return {}

    value = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(value, dict):
        return {}

    return value


def write_json(
    path: Path,
    value: dict[str, Any],
) -> None:
    """Write a JSON object."""
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


def nanny_columns(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Keep only columns required by CBPE."""
    required = [
        TARGET_COLUMN,
        PREDICTION_COLUMN,
        PROBABILITY_COLUMN,
    ]

    missing = (
        set(required)
        - set(frame.columns)
    )

    if missing:
        raise ValueError(
            "Missing NannyML columns: "
            + ", ".join(sorted(missing))
        )

    return frame[required].copy()


def contract_test_recall() -> float | None:
    """Read champion test recall from the model hand-off file."""
    contract = read_json(
        BEST_MODEL_PATH
    )

    metrics = contract.get("metrics")

    if not isinstance(metrics, dict):
        return None

    value = metrics.get(
        "test_recall"
    )

    if value is None:
        return None

    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare champion predictions "
            "for isolated NannyML execution."
        )
    )

    parser.add_argument(
        "--period-dir",
        type=Path,
        default=DEFAULT_PERIOD_DIR,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        model,
        model_threshold,
        model_version,
    ) = load_champion()

    period_frames: dict[
        str,
        pd.DataFrame,
    ] = {}

    for period, filename in (
        PERIOD_FILES.items()
    ):
        source_path = (
            args.period_dir
            / filename
        )

        frame = load_frame(
            source_path
        )

        predicted = add_predictions(
            frame,
            model,
            model_threshold,
        )

        output = nanny_columns(
            predicted
        )

        output_path = (
            args.output_dir
            / f"{period}.csv"
        )

        output.to_csv(
            output_path,
            index=False,
        )

        period_frames[period] = (
            output
        )

        print(
            f"Prepared {period}: "
            f"{output_path}"
        )

    reference = pd.concat(
        [
            period_frames["period_1"],
            period_frames["period_2"],
        ],
        ignore_index=True,
    )

    reference_path = (
        args.output_dir
        / "reference.csv"
    )

    reference.to_csv(
        reference_path,
        index=False,
    )

    metadata = {
        "model_threshold": (
            float(model_threshold)
        ),
        "model_version": str(
            model_version
        ),
        "contract_test_recall": (
            contract_test_recall()
        ),
        "reference_periods": [
            "period_1",
            "period_2",
        ],
        "columns": {
            "target": TARGET_COLUMN,
            "prediction": (
                PREDICTION_COLUMN
            ),
            "probability": (
                PROBABILITY_COLUMN
            ),
        },
    }

    write_json(
        args.output_dir
        / "metadata.json",
        metadata,
    )

    print()
    print(
        "NannyML inputs prepared successfully."
    )
    print(
        f"Model version: {model_version}"
    )
    print(
        f"Model threshold: {model_threshold}"
    )
    print(
        f"Output directory: {args.output_dir}"
    )


if __name__ == "__main__":
    main()