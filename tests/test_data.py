import pandas as pd

from fraud.data import time_split


def make_df(n_rows=100):
    return pd.DataFrame(
        {
            "Time": list(range(n_rows - 1, -1, -1)),
            "V1": [0.0] * n_rows,
            "Class": [0] * n_rows,
        }
    )


def test_time_split_sorts_by_time():
    df = make_df()

    train, val, test = time_split(df)

    assert train["Time"].is_monotonic_increasing
    assert val["Time"].is_monotonic_increasing
    assert test["Time"].is_monotonic_increasing


def test_time_split_uses_60_20_20_ratio():
    df = make_df()

    train, val, test = time_split(df)

    assert len(train) == 60
    assert len(val) == 20
    assert len(test) == 20