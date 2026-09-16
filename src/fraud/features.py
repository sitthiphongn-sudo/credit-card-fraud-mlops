"""การแปลงข้อมูลชุดเดียว ใช้ทั้งตอนเทรนและตอนให้บริการ (กัน Training-Serving Skew) — ผู้รับผิดชอบ: นาคินทร์

transformer ถูกบันทึกรวมอยู่ใน sklearn Pipeline เดียวกับโมเดล
API จึงรับข้อมูลดิบแล้วแปลงด้วยค่าที่ fit จากชุดฝึกเสมอ
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from fraud.validate import V_COLS


class HourOfDay(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        t = np.asarray(X, dtype=float).reshape(-1)
        hour = (t % 86_400) / 3_600
        return np.c_[np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)]

    def get_feature_names_out(self, input_features=None):
        return np.array(["hour_sin", "hour_cos"])


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("amount", Pipeline([("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                                 ("scale", StandardScaler())]), ["Amount"]),
            ("hour", HourOfDay(), ["Time"]),
            ("v", StandardScaler(), V_COLS),
        ],
        remainder="drop",
    )


def build_pipeline(model) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("model", model)])


def split_xy(df: pd.DataFrame, target: str = "Class"):
    return df.drop(columns=[target]), df[target].astype(int)
