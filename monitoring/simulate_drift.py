"""Create four simulated production periods for monitoring.

Period 1-2:
    Normal production windows.

Period 3:
    Data drift. Change Amount, V14 and V17 while preserving labels.

Period 4:
    Concept drift. Keep every model feature unchanged while changing labels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from common import (
    DEFAULT_PERIOD_DIR,
    FEATURE_COLUMNS,
    ROOT,
    TARGET_COLUMN,
    load_frame,
    load_params,
    write_json,
)

NUMBER_OF_PERIODS = 4


def get_simulation_settings(params: dict) -> dict[str, float]:
    """Read simulation settings from configs/params.yaml."""
    monitoring = params.get("monitoring")

    if not isinstance(monitoring, dict):
        raise ValueError(
            "Missing 'monitoring' section in configs/params.yaml"
        )

    simulation = monitoring.get("simulation")

    if not isinstance(simulation, dict):
        raise ValueError(
            "Missing 'monitoring.simulation' section in configs/params.yaml"
        )

    required = {
        "amount_multiplier",
        "v14_shift_mean",
        "v17_shift_mean",
        "shift_std",
        "small_amount_quantile",
        "concept_flip_share",
    }

    missing = required - simulation.keys()

    if missing:
        raise ValueError(
            "Missing monitoring.simulation parameters: "
            + ", ".join(sorted(missing))
        )

    settings = {
        key: float(simulation[key])
        for key in required
    }

    if settings["amount_multiplier"] <= 0:
        raise ValueError(
            "amount_multiplier must be greater than zero"
        )

    if settings["shift_std"] < 0:
        raise ValueError(
            "shift_std must be zero or greater"
        )

    if not 0 < settings["small_amount_quantile"] < 1:
        raise ValueError(
            "small_amount_quantile must be between 0 and 1"
        )

    if not 0 < settings["concept_flip_share"] <= 1:
        raise ValueError(
            "concept_flip_share must be between 0 and 1"
        )

    return settings


def data_drift(
    df: pd.DataFrame,
    *,
    seed: int,
    settings: dict[str, float],
) -> pd.DataFrame:
    """Inject data drift while keeping labels unchanged."""
    out = df.copy()

    rng = np.random.default_rng(seed)

    out["Amount"] = (
        out["Amount"] * settings["amount_multiplier"]
    )

    out["V14"] = out["V14"] + rng.normal(
        settings["v14_shift_mean"],
        settings["shift_std"],
        len(out),
    )

    out["V17"] = out["V17"] + rng.normal(
        settings["v17_shift_mean"],
        settings["shift_std"],
        len(out),
    )

    return out


def concept_drift(
    df: pd.DataFrame,
    *,
    seed: int,
    small_amount_threshold: float,
    flip_share: float,
) -> pd.DataFrame:
    """Flip labels for low-amount normal transactions.

    No model feature is modified.
    """
    out = df.copy()

    eligible = (
        (out["Amount"] <= small_amount_threshold)
        & (out[TARGET_COLUMN] == 0)
    )

    positions = np.flatnonzero(
        eligible.to_numpy()
    )

    if len(positions) == 0:
        raise ValueError(
            "No eligible low-amount normal rows "
            "are available for concept drift"
        )

    count = max(
        1,
        int(np.ceil(len(positions) * flip_share)),
    )

    rng = np.random.default_rng(seed)

    selected = rng.choice(
        positions,
        size=count,
        replace=False,
    )

    class_index = out.columns.get_loc(
        TARGET_COLUMN
    )

    out.iloc[selected, class_index] = 1

    return out


def split_periods(
    df: pd.DataFrame,
) -> list[pd.DataFrame]:
    """Sort by Time and split into four consecutive windows."""
    if len(df) < NUMBER_OF_PERIODS:
        raise ValueError(
            "Need at least four rows "
            "to create four production periods"
        )

    ordered = (
        df.sort_values(
            "Time",
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    indices = np.array_split(
        np.arange(len(ordered)),
        NUMBER_OF_PERIODS,
    )

    return [
        ordered.iloc[index]
        .copy()
        .reset_index(drop=True)
        for index in indices
    ]


def verify_simulation(
    period_3_base: pd.DataFrame,
    period_3: pd.DataFrame,
    period_4_base: pd.DataFrame,
    period_4: pd.DataFrame,
) -> None:
    """Verify that each drift scenario changes only what it should."""
    if not period_3_base[TARGET_COLUMN].equals(
        period_3[TARGET_COLUMN]
    ):
        raise RuntimeError(
            "Period 3 changed labels. "
            "Data drift must change features only."
        )

    if not period_4_base[
        FEATURE_COLUMNS
    ].equals(
        period_4[FEATURE_COLUMNS]
    ):
        raise RuntimeError(
            "Period 4 changed model features. "
            "Concept drift must keep features unchanged."
        )

    changed_back = (
        (period_4_base[TARGET_COLUMN] == 1)
        & (period_4[TARGET_COLUMN] == 0)
    ).sum()

    changed_to_fraud = (
        (period_4_base[TARGET_COLUMN] == 0)
        & (period_4[TARGET_COLUMN] == 1)
    ).sum()

    if changed_back:
        raise RuntimeError(
            "Period 4 changed fraud labels back to normal"
        )

    if not changed_to_fraud:
        raise RuntimeError(
            "Period 4 did not flip any labels to fraud"
        )


def build_summary(
    source_path: Path,
    periods: dict[str, pd.DataFrame],
    period_3_base: pd.DataFrame,
    period_4_base: pd.DataFrame,
    small_amount_threshold: float,
    seed: int,
    settings: dict[str, float],
) -> dict:
    """Build an audit record of the simulated changes."""
    period_3 = periods[
        "period_3_data_drift"
    ]

    period_4 = periods[
        "period_4_concept_drift"
    ]

    labels_flipped = int(
        (
            (period_4_base[TARGET_COLUMN] == 0)
            & (period_4[TARGET_COLUMN] == 1)
        ).sum()
    )

    return {
        "source": str(source_path),
        "seed": seed,
        "simulation": {
            **settings,
            "small_amount_threshold": (
                small_amount_threshold
            ),
        },
        "period_rows": {
            name: len(frame)
            for name, frame in periods.items()
        },
        "reference_files": {
            "normal_reference": "period_1.csv",
            "period_3_control": (
                "period_3_reference.csv"
            ),
            "period_4_control": (
                "period_4_reference.csv"
            ),
        },
        "period_3_data_drift": {
            "amount_mean_before": float(
                period_3_base["Amount"].mean()
            ),
            "amount_mean_after": float(
                period_3["Amount"].mean()
            ),
            "v14_mean_before": float(
                period_3_base["V14"].mean()
            ),
            "v14_mean_after": float(
                period_3["V14"].mean()
            ),
            "v17_mean_before": float(
                period_3_base["V17"].mean()
            ),
            "v17_mean_after": float(
                period_3["V17"].mean()
            ),
            "labels_changed": int(
                (
                    period_3_base[TARGET_COLUMN]
                    != period_3[TARGET_COLUMN]
                ).sum()
            ),
        },
        "period_4_concept_drift": {
            "features_unchanged": bool(
                period_4_base[
                    FEATURE_COLUMNS
                ].equals(
                    period_4[
                        FEATURE_COLUMNS
                    ]
                )
            ),
            "fraud_rate_before": float(
                period_4_base[
                    TARGET_COLUMN
                ].mean()
            ),
            "fraud_rate_after": float(
                period_4[
                    TARGET_COLUMN
                ].mean()
            ),
            "labels_flipped_to_fraud": (
                labels_flipped
            ),
        },
    }


def resolve_input(
    requested: Path,
) -> Path:
    """Prefer processed test data.

    Fall back to the committed validation sample
    only when processed test data does not exist.
    """
    if requested.exists():
        return requested

    sample = (
        ROOT
        / "data"
        / "sample"
        / "valid.csv"
    )

    expected_test = (
        ROOT
        / "data"
        / "processed"
        / "test.csv"
    )

    if (
        requested == expected_test
        and sample.exists()
    ):
        print(
            f"WARNING: {requested} is missing; "
            f"using demo sample {sample}"
        )
        return sample

    raise FileNotFoundError(
        f"Input data not found: {requested}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create four simulated "
            "production periods."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=(
            ROOT
            / "data"
            / "processed"
            / "test.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PERIOD_DIR,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Override the project seed "
            "for debugging only."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    params = load_params()

    seed = (
        int(params["seed"])
        if args.seed is None
        else args.seed
    )

    settings = get_simulation_settings(
        params
    )

    input_path = resolve_input(
        args.input
    )

    source = load_frame(
        input_path
    )

    small_amount_threshold = float(
        source["Amount"].quantile(
            settings[
                "small_amount_quantile"
            ]
        )
    )

    (
        period_1,
        period_2,
        period_3_base,
        period_4_base,
    ) = split_periods(source)

    period_3 = data_drift(
        period_3_base,
        seed=seed,
        settings=settings,
    )

    period_4 = concept_drift(
        period_4_base,
        seed=seed,
        small_amount_threshold=(
            small_amount_threshold
        ),
        flip_share=settings[
            "concept_flip_share"
        ],
    )

    periods = {
        "period_1": period_1,
        "period_2": period_2,
        "period_3_data_drift": period_3,
        "period_4_concept_drift": period_4,
    }

    verify_simulation(
        period_3_base,
        period_3,
        period_4_base,
        period_4,
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Full source is kept for audit only.
    source.to_csv(
        args.output_dir / "reference.csv",
        index=False,
    )

    # Normal production windows.
    period_1.to_csv(
        args.output_dir / "period_1.csv",
        index=False,
    )

    period_2.to_csv(
        args.output_dir / "period_2.csv",
        index=False,
    )

    # Matched controls BEFORE drift injection.
    period_3_base.to_csv(
        args.output_dir
        / "period_3_reference.csv",
        index=False,
    )

    period_4_base.to_csv(
        args.output_dir
        / "period_4_reference.csv",
        index=False,
    )

    # Injected scenarios.
    period_3.to_csv(
        args.output_dir
        / "period_3_data_drift.csv",
        index=False,
    )

    period_4.to_csv(
        args.output_dir
        / "period_4_concept_drift.csv",
        index=False,
    )

    summary = build_summary(
        input_path,
        periods,
        period_3_base,
        period_4_base,
        small_amount_threshold,
        seed,
        settings,
    )

    write_json(
        args.output_dir
        / "simulation_summary.json",
        summary,
    )

    print(
        "Created production simulation in "
        f"{args.output_dir}"
    )

    print(
        "Period 3 changed labels: "
        f"{summary['period_3_data_drift']['labels_changed']}"
    )

    print(
        "Period 4 labels flipped to fraud: "
        f"{summary['period_4_concept_drift']['labels_flipped_to_fraud']}"
    )


if __name__ == "__main__":
    main()