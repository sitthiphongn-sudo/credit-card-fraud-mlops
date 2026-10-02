from pathlib import Path

import pandas as pd
import pandera.pandas as pa
import pytest

from fraud.validate import validate,validate_raw


SAMPLE_DIR = Path("data/sample")


def test_valid_sample_passes_validation():
    df = pd.read_csv(SAMPLE_DIR / "valid.csv")

    result = validate(df)

    assert len(result) == len(df)


@pytest.mark.parametrize(
    "filename",
    [
        "bad_missing_value.csv",
        "bad_wrong_type.csv",
        "bad_negative_amount.csv",
        "bad_missing_column.csv",
        "bad_out_of_range.csv",
    ],
)
def test_invalid_samples_fail_validation(filename):
    df = pd.read_csv(SAMPLE_DIR / filename)

    with pytest.raises(pa.errors.SchemaErrors):
        validate(df)

def test_validate_raw_allows_statistical_outlier():
    df = pd.read_csv(SAMPLE_DIR / "valid.csv").head(1).copy()

    df.loc[df.index[0], "V1"] = 1000.0

    result = validate_raw(df)

    assert len(result) == 1