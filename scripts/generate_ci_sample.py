"""Create reproducible, clearly synthetic CI fixtures (never used as production data)."""

from pathlib import Path

import numpy as np
import pandas as pd


def generate(directory: Path = Path("data/sample"), rows: int = 2000) -> None:
    rng = np.random.default_rng(413008)
    directory.mkdir(parents=True, exist_ok=True)
    labels = np.tile([0] * 9 + [1], rows // 10 + 1)[:rows]
    frame = pd.DataFrame(rng.normal(size=(rows, 28)), columns=[f"V{i}" for i in range(1, 29)])
    frame.insert(0, "Time", np.arange(rows, dtype=float) * 60)
    frame["V14"] = rng.normal(loc=-3 * labels, scale=0.8)
    frame["V17"] = rng.normal(loc=-2 * labels, scale=0.8)
    frame["Amount"] = rng.gamma(shape=2, scale=30, size=rows)
    frame["Class"] = labels
    frame.to_csv(directory / "good.csv", index=False)
    bad = frame.copy()
    bad.loc[0, "Amount"] = -10
    bad.to_csv(directory / "bad.csv", index=False)


if __name__ == "__main__":
    generate()
