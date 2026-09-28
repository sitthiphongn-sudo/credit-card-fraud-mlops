"""
Unit tests ของโมดูล ``fraud.features``

ครอบคลุม 3 ข้อที่โจทย์กำหนดไว้ ("ทดสอบว่ากันการรั่ว/เพี้ยนของข้อมูลได้จริง"):
    (ก) ผลลัพธ์จาก DataFrame กับจาก JSON ทีละรายการต้องเท่ากัน
        -> test_dataframe_and_single_json_record_give_same_result
    (ข) สลับลำดับคอลัมน์แล้วผลต้องไม่เปลี่ยน
        -> test_column_order_does_not_affect_result
    (ค) โหลดโมเดลที่บันทึกด้วย joblib แล้วทำนาย/แปลงข้อมูลได้ค่าเดิม
        -> test_joblib_roundtrip_gives_identical_transform
        -> test_joblib_roundtrip_gives_identical_predictions_end_to_end

นอกจากนั้นยังมี test เสริมที่พิสูจน์ตรรกะของแต่ละฟีเจอร์โดยตรง (cyclical time
encoding, การลด skew ของ Amount, การ fit scaler จากชุดฝึกเท่านั้น) เพื่อให้
มั่นใจได้ว่าไม่ใช่แค่ "รันได้" แต่ "ถูกต้องตามหลักการ" ด้วย
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import skew

from fraud.features import (
    OUTPUT_COLUMNS,
    V_COLUMNS,
    FraudFeatureTransformer,
    encode_time_cyclical,
    load_transformer,
    save_transformer,
)


def _make_synthetic_df(n: int = 500, seed: int = 0, fraud_ratio: float = 0.02) -> pd.DataFrame:
    """สร้างข้อมูลจำลองหน้าตาเหมือนชุดข้อมูลจริง (Time, V1-V28, Amount, Class)
    โดยตั้งใจให้ Amount เบ้ขวาแรง (คล้ายของจริง) เพื่อทดสอบการลด skew ได้จริง
    """
    rng = np.random.default_rng(seed)
    time = np.sort(rng.integers(0, 172_800, size=n))  # ~2 วัน หน่วยวินาที
    v_data = {f"V{i}": rng.normal(loc=0.0, scale=3.0, size=n) for i in range(1, 29)}
    # ธุรกรรมส่วนใหญ่ยอดเล็ก (exponential) มีบางรายการยอดใหญ่มาก (เบ้ขวาแรง)
    amount = rng.exponential(scale=50.0, size=n) + rng.choice(
        [0.0, 5000.0], size=n, p=[0.995, 0.005]
    )
    fraud_class = rng.choice([0, 1], size=n, p=[1 - fraud_ratio, fraud_ratio])
    return pd.DataFrame({"Time": time, **v_data, "Amount": amount, "Class": fraud_class})


@pytest.fixture
def train_df() -> pd.DataFrame:
    return _make_synthetic_df(n=800, seed=1)


@pytest.fixture
def test_df() -> pd.DataFrame:
    # seed ต่างกันโดยตั้งใจ เพื่อให้ mean/std ของชุดนี้ต่างจากชุดฝึกจริง ๆ
    # (ถ้า scaler ดันไป fit จากชุดนี้โดยไม่ตั้งใจ ค่าที่ได้จะต่างจากที่คาดไว้ชัดเจน)
    return _make_synthetic_df(n=200, seed=2)


@pytest.fixture
def fitted_transformer(train_df) -> FraudFeatureTransformer:
    return FraudFeatureTransformer().fit(train_df)


# --------------------------------------------------------------------------
# เข้ารหัสเวลาแบบวงกลม (cyclical time encoding)
# --------------------------------------------------------------------------


def test_hour_23_and_hour_0_are_close_on_the_circle():
    """23:00 กับ 00:00 ต้องอยู่ใกล้กันบนวงกลม แม้ตัวเลขชั่วโมงต่างกันถึง 23"""
    df = pd.DataFrame({"Time": [23 * 3600, 0]})
    out = encode_time_cyclical(df)
    p23 = out.loc[0, ["hour_sin", "hour_cos"]].to_numpy(dtype=float)
    p00 = out.loc[1, ["hour_sin", "hour_cos"]].to_numpy(dtype=float)
    dist_23_00 = np.linalg.norm(p23 - p00)

    # เทียบกับ 12:00 ซึ่งควรอยู่ "ไกลที่สุด" จาก 00:00 บนวงกลม (อยู่ตรงข้ามกันพอดี)
    p12 = (
        encode_time_cyclical(pd.DataFrame({"Time": [12 * 3600]}))
        .loc[0, ["hour_sin", "hour_cos"]]
        .to_numpy(dtype=float)
    )
    dist_12_00 = np.linalg.norm(p12 - p00)

    assert dist_23_00 < dist_12_00
    # ระยะห่างบนวงกลมหนึ่งหน่วยของชั่วโมงที่ติดกันสองชั่วโมงใด ๆ ต้องเท่ากันหมด
    expected_adjacent_hour_distance = 2 * np.sin(np.pi / 24)
    assert dist_23_00 == pytest.approx(expected_adjacent_hour_distance, abs=1e-9)


def test_hour_of_day_wraps_correctly_across_multiple_days():
    # 172800 วินาที = ครบ 2 วันพอดี บวกอีก 1 ชั่วโมง -> วันที่ 3 เวลา 01:00 น.
    # hour_of_day ต้อง "วน" กลับมาเป็น 1 ไม่ใช่เดินหน้าต่อไปเรื่อย ๆ
    df = pd.DataFrame({"Time": [172_800 + 3600]})
    out = encode_time_cyclical(df)
    assert out.loc[0, "hour_of_day"] == 1


# --------------------------------------------------------------------------
# แปลง Amount ด้วย log1p แล้ว scale
# --------------------------------------------------------------------------


def test_amount_log1p_scaling_reduces_skewness(train_df, fitted_transformer):
    out = fitted_transformer.transform(train_df)
    raw_skew = skew(train_df["Amount"])
    transformed_skew = skew(out["amount_log_scaled"])
    assert abs(transformed_skew) < abs(raw_skew)


def test_amount_scaled_has_zero_mean_unit_std_on_the_set_it_was_fit_on(
    train_df, fitted_transformer
):
    out = fitted_transformer.transform(train_df)
    assert out["amount_log_scaled"].mean() == pytest.approx(0.0, abs=1e-8)
    assert out["amount_log_scaled"].std(ddof=0) == pytest.approx(1.0, abs=1e-8)


# --------------------------------------------------------------------------
# V1-V28 ต้อง fit จากชุดฝึกเท่านั้น (กัน data leakage)
# --------------------------------------------------------------------------


def test_v_scaler_uses_train_statistics_not_the_data_being_transformed(
    train_df, test_df, fitted_transformer
):
    # คำนวณค่าที่ "ควรจะได้" ด้วยมือ โดยใช้ mean/std ของชุดฝึกไปแปลงชุดทดสอบ
    train_mean = train_df["V1"].mean()
    train_std = train_df["V1"].std(ddof=0)
    expected = (test_df["V1"] - train_mean) / train_std

    out_test = fitted_transformer.transform(test_df)
    np.testing.assert_allclose(
        out_test["V1_scaled"].to_numpy(), expected.to_numpy(), atol=1e-8
    )

    # ต้อง "ไม่" ตรงกับผลที่ควรได้ถ้า scaler ไป fit จากชุดทดสอบเอง (กันไม่ให้
    # การ leak เกิดขึ้นโดยบังเอิญแล้ว test นี้ไม่จับได้)
    leaked_mean = test_df["V1"].mean()
    leaked_std = test_df["V1"].std(ddof=0)
    leaked = (test_df["V1"] - leaked_mean) / leaked_std
    assert not np.allclose(out_test["V1_scaled"].to_numpy(), leaked.to_numpy(), atol=1e-8)


# --------------------------------------------------------------------------
# (ก) DataFrame vs. JSON ทีละรายการ ต้องให้ผลเท่ากัน
# --------------------------------------------------------------------------


def test_dataframe_and_single_json_record_give_same_result(train_df, fitted_transformer):
    out_batch = fitted_transformer.transform(train_df)

    # จำลองสิ่งที่ serving/app.py ทำจริง: รับ request เป็น JSON ทีละ 1 รายการ
    row0 = train_df.iloc[0]
    record = json.loads(row0.to_json())

    out_single = fitted_transformer.transform(record)

    pd.testing.assert_frame_equal(
        out_batch.iloc[[0]].reset_index(drop=True),
        out_single.reset_index(drop=True),
        atol=1e-8,
    )


# --------------------------------------------------------------------------
# (ข) สลับลำดับคอลัมน์ของ input ผลลัพธ์ต้องไม่เปลี่ยน
# --------------------------------------------------------------------------


def test_column_order_does_not_affect_result(train_df, fitted_transformer):
    out_original = fitted_transformer.transform(train_df)

    shuffled_columns = list(train_df.columns)[::-1]
    df_shuffled = train_df[shuffled_columns]
    out_shuffled = fitted_transformer.transform(df_shuffled)

    pd.testing.assert_frame_equal(out_original, out_shuffled, atol=1e-8)


def test_output_column_order_is_always_fixed(train_df, fitted_transformer):
    out = fitted_transformer.transform(train_df)
    assert list(out.columns) == OUTPUT_COLUMNS
    assert len(OUTPUT_COLUMNS) == 3 + len(V_COLUMNS)  # hour_sin/cos + amount + 28 V


# --------------------------------------------------------------------------
# (ค) โหลดด้วย joblib แล้วต้องได้ผลเดิมทุกประการ
# --------------------------------------------------------------------------


def test_joblib_roundtrip_gives_identical_transform(train_df, test_df, fitted_transformer, tmp_path):
    out_before = fitted_transformer.transform(test_df)

    path = tmp_path / "feature_transformer.joblib"
    save_transformer(fitted_transformer, path)
    loaded = load_transformer(path)

    out_after = loaded.transform(test_df)
    pd.testing.assert_frame_equal(out_before, out_after)


def test_joblib_roundtrip_gives_identical_predictions_end_to_end(train_df, test_df, tmp_path):
    """ทดสอบระดับ pipeline เต็มรูปแบบ (transformer + โมเดล) ให้ใกล้เคียงของจริง
    ที่สุด: เทรน -> บันทึกด้วย joblib -> โหลดกลับมา -> ทำนาย ต้องได้ค่าเดิมเป๊ะ
    """
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    X_train = train_df.drop(columns=["Class"])
    y_train = train_df["Class"]
    X_test = test_df.drop(columns=["Class"])

    pipeline = Pipeline(
        [
            ("features", FraudFeatureTransformer()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    pipeline.fit(X_train, y_train)
    preds_before = pipeline.predict_proba(X_test)

    path = tmp_path / "fraud_pipeline.joblib"
    joblib.dump(pipeline, path)
    loaded_pipeline = joblib.load(path)
    preds_after = loaded_pipeline.predict_proba(X_test)

    np.testing.assert_allclose(preds_before, preds_after)


# --------------------------------------------------------------------------
# เคสผิดพลาดที่ควรมี error ชัดเจน
# --------------------------------------------------------------------------


def test_transform_before_fit_raises_clear_error(train_df):
    transformer = FraudFeatureTransformer()
    with pytest.raises(RuntimeError, match="fit"):
        transformer.transform(train_df)


def test_missing_required_column_raises_clear_error(train_df, fitted_transformer):
    broken = train_df.drop(columns=["V5"])
    with pytest.raises(ValueError, match="V5"):
        fitted_transformer.transform(broken)


def test_negative_amount_raises_clear_error(train_df, fitted_transformer):
    broken = train_df.copy()
    broken.loc[0, "Amount"] = -10.0
    with pytest.raises(ValueError, match="Amount"):
        fitted_transformer.transform(broken)

def test_nan_in_dataframe_raises_clear_error(train_df, fitted_transformer):
    broken = train_df.copy()
    broken.loc[3, "V7"] = np.nan
    with pytest.raises(ValueError, match="V7"):
        fitted_transformer.transform(broken)


def test_null_in_single_json_record_raises_clear_error(train_df, fitted_transformer):
    # จำลอง request ที่ส่ง null มาใน JSON (ฝั่งให้บริการ)
    record = json.loads(train_df.iloc[0].to_json())
    record["Amount"] = None
    with pytest.raises(ValueError, match="Amount"):
        fitted_transformer.transform(record)


def test_fit_rejects_nan_in_training_data(train_df):
    broken = train_df.copy()
    broken.loc[0, "Time"] = np.nan
    with pytest.raises(ValueError, match="Time"):
        FraudFeatureTransformer().fit(broken)
