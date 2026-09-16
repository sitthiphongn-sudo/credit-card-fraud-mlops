import numpy as np
import pandas as pd
import pytest
from pandera.errors import SchemaErrors
from sklearn.linear_model import LogisticRegression

from fraud.features import build_pipeline
from fraud.validate import FEATURES, V_COLS, validate


def make_df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, 28)), columns=V_COLS)
    df.insert(0, "Time", np.arange(n, dtype=float) * 100)
    df["Amount"] = rng.gamma(2, 30, n)
    df["Class"] = (rng.random(n) < 0.1).astype(int)
    return df


def test_valid_data_passes():
    validate(make_df())


def test_bad_data_stops():
    bad = make_df()
    bad.loc[0, "Amount"] = -10
    bad.loc[1, "V3"] = np.nan
    with pytest.raises(SchemaErrors):
        validate(bad)


def test_same_transform_train_and_serve():
    df = make_df()
    pipe = build_pipeline(LogisticRegression(max_iter=500)).fit(df[FEATURES], df["Class"])
    a = pipe.predict_proba(df[FEATURES].head(5))[:, 1]
    b = pipe.predict_proba(pd.DataFrame(df[FEATURES].head(5).to_dict("records")))[:, 1]
    assert np.allclose(a, b)
