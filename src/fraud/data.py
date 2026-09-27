"""Load raw data, split by time, and create data version metadata."""

import hashlib
import json
from pathlib import Path

import pandas as pd

from fraud.config import ROOT, load_params


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    p = load_params()
    return pd.read_csv(path or ROOT / p["data"]["raw_path"])


def time_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sort by Time and split into train/validation/test sets."""
    s = load_params()["data"]["split"]

    df = df.sort_values("Time", kind="mergesort").reset_index(drop=True)

    n = len(df)
    a = int(n * s["train"])
    b = int(n * (s["train"] + s["val"]))

    train = df.iloc[:a].copy()
    val = df.iloc[a:b].copy()
    test = df.iloc[b:].copy()

    return train, val, test


def split_summary(df: pd.DataFrame) -> dict:
    """Return row count and Time range for one split."""
    return {
        "rows": len(df),
        "time_min": float(df["Time"].min()),
        "time_max": float(df["Time"].max()),
    }


def prepare_data() -> None:
    """Create processed train/val/test files and data_version.json."""
    params = load_params()

    raw_path = ROOT / params["data"]["raw_path"]
    processed_dir = ROOT / params["data"]["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw(raw_path)

    train, val, test = time_split(df)

    train_rows_before = len(train)
    train = train.drop_duplicates().reset_index(drop=True)
    train_duplicates_removed = train_rows_before - len(train)

    train_path = processed_dir / "train.csv"
    val_path = processed_dir / "val.csv"
    test_path = processed_dir / "test.csv"

    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    test.to_csv(test_path, index=False)

    metadata = {
        "raw_file": str(raw_path.relative_to(ROOT)),
        "raw_hash": file_hash(raw_path),
        "raw_rows": len(df),
        "train_duplicates_removed": train_duplicates_removed,
        "splits": {
            "train": split_summary(train),
            "val": split_summary(val),
            "test": split_summary(test),
        },
    }

    version_path = processed_dir / "data_version.json"

    with open(version_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Raw rows: {len(df):,}")
    print(f"Train rows: {len(train):,}")
    print(f"Validation rows: {len(val):,}")
    print(f"Test rows: {len(test):,}")
    print(f"Train duplicates removed: {train_duplicates_removed:,}")
    print(f"Raw hash: {metadata['raw_hash']}")
    print(f"Saved metadata: {version_path}")


if __name__ == "__main__":
    prepare_data()