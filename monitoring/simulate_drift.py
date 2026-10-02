"""Create four simulated production periods for the monitoring demonstration.

Owner: Chotikan

Period 1-2: normal data.
Period 3: data drift by changing Amount, V14 and V17 only.
Period 4: concept drift by keeping X unchanged and flipping labels for a small-amount segment.
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

# These values control the *simulation severity*, not production alert thresholds.
# They are kept in one place and are also written to simulation_summary.json for auditability.
AMOUNT_MULTIPLIER = 1.8
SHIFT_MEAN = 1.5
SHIFT_STD = 0.3
SMALL_AMOUNT_QUANTILE = 0.25
CONCEPT_FLIP_SHARE = 0.05
NUMBER_OF_PERIODS = 4


def data_drift(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Change feature distributions while preserving labels exactly."""
    out = df.copy()
    rng = np.random.default_rng(seed)

    out["Amount"] = out["Amount"] * AMOUNT_MULTIPLIER
    out["V14"] = out["V14"] + rng.normal(SHIFT_MEAN, SHIFT_STD, len(out))
    out["V17"] = out["V17"] + rng.normal(SHIFT_MEAN, SHIFT_STD, len(out))
    return out


def concept_drift(
    df: pd.DataFrame,
    *,
    seed: int,
    small_amount_threshold: float,
) -> pd.DataFrame:
    """Flip some low-amount normal labels to fraud without changing any model feature."""
    out = df.copy()
    eligible = (out["Amount"] <= small_amount_threshold) & (out[TARGET_COLUMN] == 0)
    positions = np.flatnonzero(eligible.to_numpy())

    if len(positions) == 0:
        raise ValueError("No eligible low-amount normal rows are available for concept drift")

    count = max(1, int(np.ceil(len(positions) * CONCEPT_FLIP_SHARE)))
    rng = np.random.default_rng(seed)
    selected = rng.choice(positions, size=count, replace=False)
    class_index = out.columns.get_loc(TARGET_COLUMN)
    out.iloc[selected, class_index] = 1
    return out


def split_periods(df: pd.DataFrame) -> list[pd.DataFrame]:
    """Sort by Time and split into four consecutive production windows."""
    if len(df) < NUMBER_OF_PERIODS:
        raise ValueError("Need at least four rows to create four production periods")

    ordered = df.sort_values("Time", kind="mergesort").reset_index(drop=True)
    indices = np.array_split(np.arange(len(ordered)), NUMBER_OF_PERIODS)
    return [ordered.iloc[index].copy().reset_index(drop=True) for index in indices]


def verify_simulation(
    period_3_base: pd.DataFrame,
    period_3: pd.DataFrame,
    period_4_base: pd.DataFrame,
    period_4: pd.DataFrame,
) -> None:
    """Fail fast if the injected drift does not match its intended definition."""
    if not period_3_base[TARGET_COLUMN].equals(period_3[TARGET_COLUMN]):
        raise RuntimeError("Period 3 changed labels; data drift must change X only")

    if not period_4_base[FEATURE_COLUMNS].equals(period_4[FEATURE_COLUMNS]):
        raise RuntimeError("Period 4 changed model features; concept drift must keep X unchanged")

    changed_back = ((period_4_base[TARGET_COLUMN] == 1) & (period_4[TARGET_COLUMN] == 0)).sum()
    changed_to_fraud = ((period_4_base[TARGET_COLUMN] == 0) & (period_4[TARGET_COLUMN] == 1)).sum()
    if changed_back:
        raise RuntimeError("Period 4 changed fraud labels back to normal")
    if not changed_to_fraud:
        raise RuntimeError("Period 4 did not flip any labels to fraud")


def build_summary(
    source_path: Path,
    periods: dict[str, pd.DataFrame],
    period_3_base: pd.DataFrame,
    period_4_base: pd.DataFrame,
    small_amount_threshold: float,
    seed: int,
) -> dict:
    """Build an audit record showing exactly what was changed."""
    period_3 = periods["period_3_data_drift"]
    period_4 = periods["period_4_concept_drift"]
    labels_flipped = int(
        ((period_4_base[TARGET_COLUMN] == 0) & (period_4[TARGET_COLUMN] == 1)).sum()
    )

    return {
        "source": str(source_path),
        "seed": seed,
        "simulation": {
            "amount_multiplier": AMOUNT_MULTIPLIER,
            "v14_v17_shift_mean": SHIFT_MEAN,
            "v14_v17_shift_std": SHIFT_STD,
            "small_amount_quantile": SMALL_AMOUNT_QUANTILE,
            "small_amount_threshold": small_amount_threshold,
            "concept_flip_share": CONCEPT_FLIP_SHARE,
        },
        "period_rows": {name: len(frame) for name, frame in periods.items()},
        "period_3_data_drift": {
            "amount_mean_before": float(period_3_base["Amount"].mean()),
            "amount_mean_after": float(period_3["Amount"].mean()),
            "v14_mean_before": float(period_3_base["V14"].mean()),
            "v14_mean_after": float(period_3["V14"].mean()),
            "v17_mean_before": float(period_3_base["V17"].mean()),
            "v17_mean_after": float(period_3["V17"].mean()),
            "labels_changed": int((period_3_base[TARGET_COLUMN] != period_3[TARGET_COLUMN]).sum()),
        },
        "period_4_concept_drift": {
            "features_unchanged": bool(period_4_base[FEATURE_COLUMNS].equals(period_4[FEATURE_COLUMNS])),
            "fraud_rate_before": float(period_4_base[TARGET_COLUMN].mean()),
            "fraud_rate_after": float(period_4[TARGET_COLUMN].mean()),
            "labels_flipped_to_fraud": labels_flipped,
        },
    }


def resolve_input(requested: Path) -> Path:
    """Prefer processed test data; use the committed valid sample only for local/demo setup."""
    if requested.exists():
        return requested

    sample = ROOT / "data" / "sample" / "valid.csv"
    if requested == ROOT / "data" / "processed" / "test.csv" and sample.exists():
        print(f"WARNING: {requested} is missing; using demo sample {sample}")
        return sample
    raise FileNotFoundError(f"Input data not found: {requested}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create four simulated production periods.")
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "processed" / "test.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PERIOD_DIR)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the project seed for debugging only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    params = load_params()
    seed = int(params["seed"]) if args.seed is None else args.seed
    input_path = resolve_input(args.input)
    source = load_frame(input_path)

    small_amount_threshold = float(source["Amount"].quantile(SMALL_AMOUNT_QUANTILE))
    period_1, period_2, period_3_base, period_4_base = split_periods(source)
    period_3 = data_drift(period_3_base, seed=seed)
    period_4 = concept_drift(
        period_4_base,
        seed=seed,
        small_amount_threshold=small_amount_threshold,
    )

    periods = {
        "period_1": period_1,
        "period_2": period_2,
        "period_3_data_drift": period_3,
        "period_4_concept_drift": period_4,
    }
    verify_simulation(period_3_base, period_3, period_4_base, period_4)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source.to_csv(args.output_dir / "reference.csv", index=False)
    for name, frame in periods.items():
        frame.to_csv(args.output_dir / f"{name}.csv", index=False)

    summary = build_summary(
        input_path,
        periods,
        period_3_base,
        period_4_base,
        small_amount_threshold,
        seed,
    )
    write_json(args.output_dir / "simulation_summary.json", summary)

    print(f"Created production simulation in {args.output_dir}")
    print(f"Period 3 changed labels: {summary['period_3_data_drift']['labels_changed']}")
    print(
        "Period 4 labels flipped to fraud: "
        f"{summary['period_4_concept_drift']['labels_flipped_to_fraud']}"
    )


if __name__ == "__main__":
    main()
