import json
from pathlib import Path

import pandas as pd

TRAIN_PATH = Path("data/processed/train.csv")
OUTPUT_PATH = Path("data/processed/schema_stats.json")

V_COLS = [f"V{i}" for i in range(1, 29)]


def main():
    df = pd.read_csv(TRAIN_PATH)

    stats = {
        "amount_p999": float(df["Amount"].quantile(0.999)),
        "v_ranges": {},
    }

    for col in V_COLS:
        col_min = float(df[col].min())
        col_max = float(df[col].max())

        # เพิ่มระยะเผื่อ 5% ของช่วงข้อมูล
        value_range = col_max - col_min
        margin = value_range * 0.20

        stats["v_ranges"][col] = {
            "train_min": col_min,
            "train_max": col_max,
            "schema_min": col_min - margin,
            "schema_max": col_max + margin,
        }

    OUTPUT_PATH.write_text(
        json.dumps(stats, indent=2),
        encoding="utf-8",
    )

    print(f"Amount p99.9: {stats['amount_p999']:.6f}")
    print()

    for col, values in stats["v_ranges"].items():
        print(
            f"{col}: "
            f"{values['schema_min']:.6f} "
            f"to {values['schema_max']:.6f}"
        )

    print()
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()