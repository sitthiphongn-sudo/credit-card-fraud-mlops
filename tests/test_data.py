import pandas as pd

from fraud.data import file_hash, time_split


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

def test_time_split_removes_duplicates_only_from_train():
    df = make_df(100)

    # Put one exact duplicate inside the training portion.
    duplicate_row = df.iloc[[90]].copy()
    df = pd.concat([df, duplicate_row], ignore_index=True)

    train, val, test = time_split(df)

    assert train.duplicated().sum() == 0


def test_time_split_has_non_overlapping_time_ranges():
    df = make_df(100)

    train, val, test = time_split(df)

    assert train["Time"].max() <= val["Time"].min()
    assert val["Time"].max() <= test["Time"].min()

def test_file_hash_is_deterministic(tmp_path):
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text("fraud-data-version", encoding="utf-8")

    first_hash = file_hash(sample_file)
    second_hash = file_hash(sample_file)

    assert first_hash == second_hash