from pathlib import Path

import pandas as pd

from fraud.validate import validate


SAMPLE_DIR = Path("data/sample")


def test_valid_sample_passes_validation():
    df = pd.read_csv(SAMPLE_DIR / "valid.csv")

    result = validate(df)

    assert len(result) == len(df)