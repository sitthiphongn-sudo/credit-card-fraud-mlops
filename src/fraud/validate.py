"""Schema ของข้อมูลและการตรวจความผิดปกติ — ผู้รับผิดชอบ: เปรมสิริวัฒณ์

ใช้ schema เดียวกันทั้งใน pipeline และใน API
ข้อมูลเสีย → raise SchemaErrors → pipeline หยุด และ log แจ้งเตือน
"""
import logging
import sys

import pandas as pd
import pandera.pandas as pa

log = logging.getLogger("fraud.validate")
V_COLS = [f"V{i}" for i in range(1, 29)]
FEATURES = ["Time", *V_COLS, "Amount"]

_feature_cols = {
    "Time": pa.Column(float, pa.Check.ge(0), nullable=False, coerce=True),
    "Amount": pa.Column(float, [pa.Check.ge(0), pa.Check.le(50_000)], nullable=False, coerce=True),
    # TODO(เปรมสิริวัฒณ์): ปรับช่วงค่า V ตามสถิติจาก EDA ของชุดฝึก
    **{c: pa.Column(float, pa.Check.in_range(-150, 150), nullable=False, coerce=True) for c in V_COLS},
}

feature_schema = pa.DataFrameSchema(_feature_cols, strict="filter")
training_schema = pa.DataFrameSchema(
    {**_feature_cols, "Class": pa.Column(int, pa.Check.isin([0, 1]), nullable=False, coerce=True)},
    strict=True,
)


def validate(df: pd.DataFrame, with_target: bool = True) -> pd.DataFrame:
    schema = training_schema if with_target else feature_schema
    try:
        return schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as err:
        cases = err.failure_cases
        log.error("DATA VALIDATION FAILED: %d problems\n%s", len(cases), cases.head(20))
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/creditcard.csv"
    try:
        validate(pd.read_csv(path))
    except pa.errors.SchemaErrors:
        sys.exit(1)
    log.info("data OK: %s", path)
