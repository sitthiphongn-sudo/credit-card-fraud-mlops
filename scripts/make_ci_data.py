"""Generate reproducible synthetic data for CI, not production model evaluation."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw/creditcard.csv"))
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--signal-strength", type=float, default=6.0)
    parser.add_argument("--metadata", type=Path, default=Path("reports/ci/data-generation.json"))
    args = parser.parse_args()
    if args.rows < 1_000:
        parser.error("--rows must be at least 1000 for time-based train/validation/test splits")
    if not np.isfinite(args.signal_strength) or args.signal_strength < 0:
        parser.error("--signal-strength must be finite and nonnegative")

    rng = np.random.default_rng(args.seed)
    labels = np.zeros(args.rows, dtype=int)
    labels[rng.choice(args.rows, size=round(args.rows * 0.02), replace=False)] = 1
    values = rng.normal(size=(args.rows, 28))
    # Removing this signal creates a negative CI demonstration with identical labels/noise.
    values[:, [13, 16]] -= labels[:, None] * args.signal_strength
    frame = pd.DataFrame(values, columns=[f"V{i}" for i in range(1, 29)])
    frame.insert(0, "Time", np.arange(args.rows, dtype=float) * 8)
    frame["Amount"] = np.round(np.minimum(rng.lognormal(3.5, 1.0, args.rows), 2_000), 2)
    frame["Class"] = labels
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, float_format="%.8f", lineterminator="\n")
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps({
        "data_kind": "synthetic CI fixture; not real-world model performance",
        "rows": args.rows,
        "fraud_rows": int(labels.sum()),
        "seed": args.seed,
        "signal_strength": args.signal_strength,
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }, indent=2), encoding="utf-8")
    print(f"Synthetic CI data: {args.rows} rows, {labels.sum()} fraud, seed={args.seed}")


if __name__ == "__main__":
    main()
