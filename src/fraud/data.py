"""Load raw data, split by time, and create data version metadata."""

import hashlib
import json
from pathlib import Path

import pandas as pd

from fraud.config import ROOT, load_params


def file_hash(path: str | Path) -> str:
    """Return a short SHA-256 hash for a file."""
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)

    return h.hexdigest()[:12]


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw credit-card dataset."""
    params = load_params()
    raw_path = path or ROOT / params["data"]["raw_path"]

    return pd.read_csv(raw_path)


def time_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Sort by Time and split into train/validation/test sets.

    Duplicate rows are removed from the training set only.
    Validation and test sets keep duplicates so evaluation better
    reflects incoming real-world data.
    """
    split_config = load_params()["data"]["split"]

    df = (
        df.sort_values("Time", kind="mergesort")
        .reset_index(drop=True)
    )

    n_rows = len(df)

    train_end = int(n_rows * split_config["train"])

    val_end = int(
        n_rows
        * (
            split_config["train"]
            + split_config["val"]
        )
    )

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    # Remove duplicates only from the training split.
    train = (
        train.drop_duplicates()
        .reset_index(drop=True)
    )

    return train, val, test


def split_summary(df: pd.DataFrame) -> dict:
    """Return row count and Time range for one split."""
    return {
        "rows": len(df),
        "time_min": float(df["Time"].min()),
        "time_max": float(df["Time"].max()),
    }


def build_data_version(
    raw_path: str | Path,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    raw_rows: int,
    train_duplicates_removed: int,
) -> dict:
    """Create metadata describing the exact dataset version."""
    raw_path = Path(raw_path)

    return {
        "raw_file": str(raw_path.relative_to(ROOT)),
        "raw_hash": file_hash(raw_path),
        "raw_rows": raw_rows,
        "train_duplicates_removed": train_duplicates_removed,
        "splits": {
            "train": split_summary(train),
            "val": split_summary(val),
            "test": split_summary(test),
        },
    }


def save_processed_data(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    metadata: dict,
) -> Path:
    """Save processed splits and data_version.json."""
    params = load_params()

    processed_dir = ROOT / params["data"]["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    train.to_csv(
        processed_dir / "train.csv",
        index=False,
    )

    val.to_csv(
        processed_dir / "val.csv",
        index=False,
    )

    test.to_csv(
        processed_dir / "test.csv",
        index=False,
    )

    version_path = processed_dir / "data_version.json"

    with open(version_path, "w", encoding="utf-8") as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    return version_path


def prepare_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict,
]:
    """
    Run the full data-preparation step.

    Returns train, validation, test, and data-version metadata.
    """
    params = load_params()

    raw_path = ROOT / params["data"]["raw_path"]

    df = load_raw(raw_path)

    raw_rows = len(df)

    # Count duplicates that belong to the training period
    # before time_split removes them.
    ordered = (
        df.sort_values("Time", kind="mergesort")
        .reset_index(drop=True)
    )

    train_end = int(
        len(ordered)
        * params["data"]["split"]["train"]
    )

    train_before = ordered.iloc[:train_end].copy()

    train_duplicates_removed = int(
        train_before.duplicated().sum()
    )

    train, val, test = time_split(df)

    metadata = build_data_version(
        raw_path=raw_path,
        train=train,
        val=val,
        test=test,
        raw_rows=raw_rows,
        train_duplicates_removed=train_duplicates_removed,
    )

    version_path = save_processed_data(
        train=train,
        val=val,
        test=test,
        metadata=metadata,
    )

    print(f"Raw rows: {raw_rows:,}")
    print(f"Train rows: {len(train):,}")
    print(f"Validation rows: {len(val):,}")
    print(f"Test rows: {len(test):,}")
    print(
        "Train duplicates removed: "
        f"{train_duplicates_removed:,}"
    )
    print(f"Raw hash: {metadata['raw_hash']}")
    print(f"Saved metadata: {version_path}")

    return train, val, test, metadata


if __name__ == "__main__":
    prepare_data()