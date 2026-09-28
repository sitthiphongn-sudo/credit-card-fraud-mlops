from pathlib import Path

import pandas as pd

from fraud.validate import AMOUNT_MAX, V_RANGES

TRAIN_PATH = Path("data/processed/train.csv")
SAMPLE_DIR = Path("data/sample")


def main():
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(TRAIN_PATH)

    valid = df[
        (df["Amount"] >= 0)
        & (df["Amount"] <= AMOUNT_MAX)
    ].copy()

    for col, (min_value, max_value) in V_RANGES.items():
        valid = valid[
            (valid[col] >= min_value)
            & (valid[col] <= max_value)
        ]

    valid = valid.head(2000).copy()

    valid.to_csv(SAMPLE_DIR / "valid.csv", index=False)

    # 1. Missing value
    bad_missing = valid.copy()
    bad_missing.loc[0, "V1"] = pd.NA
    bad_missing.to_csv(
        SAMPLE_DIR / "bad_missing_value.csv",
        index=False,
    )

    # 2. Wrong data type
    bad_type = valid.copy()
    bad_type["V2"] = bad_type["V2"].astype("object")
    bad_type.loc[0, "V2"] = "not-a-number"
    bad_type.to_csv(
        SAMPLE_DIR / "bad_wrong_type.csv",
        index=False,
    )

    # 3. Negative Amount
    bad_negative_amount = valid.copy()
    bad_negative_amount.loc[0, "Amount"] = -100
    bad_negative_amount.to_csv(
        SAMPLE_DIR / "bad_negative_amount.csv",
        index=False,
    )

    # 4. Missing column
    bad_missing_column = valid.drop(columns=["V3"])
    bad_missing_column.to_csv(
        SAMPLE_DIR / "bad_missing_column.csv",
        index=False,
    )

    # 5. Out-of-range value
    bad_out_of_range = valid.copy()
    bad_out_of_range.loc[0, "V4"] = (
        V_RANGES["V4"][1] + 100
    )
    bad_out_of_range.to_csv(
        SAMPLE_DIR / "bad_out_of_range.csv",
        index=False,
    )

    print("Created sample files:")
    for path in sorted(SAMPLE_DIR.glob("*.csv")):
        print(f"- {path}")


if __name__ == "__main__":
    main()