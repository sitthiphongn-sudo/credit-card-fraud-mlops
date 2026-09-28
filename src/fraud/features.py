"""
โมดูลแปลงข้อมูล (feature engineering) สำหรับระบบตรวจจับธุรกรรมบัตรเครดิตทุจริต

จุดสำคัญของโมดูลนี้ คือต้อง "ใช้ร่วมกัน" ได้ทั้งตอนเทรนโมเดล (train.py เรียกกับ
DataFrame เป็นก้อนใหญ่) และตอนให้บริการจริง (serving/app.py เรียกกับคำขอทีละ 1
รายการที่มาเป็น JSON) เพื่อป้องกันปัญหา "training-serving skew" คือผลลัพธ์การ
แปลงข้อมูลที่ไม่ตรงกันระหว่างสองฝั่ง ซึ่งเป็นสาเหตุบั๊กที่พบบ่อยและตรวจจับยากที่สุด
อย่างหนึ่งของระบบ ML ที่ใช้งานจริง

หลักการออกแบบ 3 ข้อ:
1. ทุก path (DataFrame หลายแถว, dict 1 แถว, list ของ dict) วิ่งผ่านฟังก์ชัน
   แปลงข้อมูลชุดเดียวกันเสมอ (ดู `_to_dataframe` และ `FraudFeatureTransformer`)
2. อ้างอิงคอลัมน์ด้วย "ชื่อ" เสมอ ไม่อ้างอิงด้วยตำแหน่ง/ลำดับ ดังนั้นสลับลำดับ
   คอลัมน์ใน input ผลลัพธ์จะไม่เปลี่ยน
3. พารามิเตอร์การ scale (ค่าเฉลี่ยและส่วนเบี่ยงเบนมาตรฐาน) ต้อง fit จากชุดฝึก
   (train) เท่านั้น ผ่าน `.fit(train_df)` แล้วนำไป `.transform()` กับชุด
   validation/test/production โดยไม่ fit ซ้ำ — ป้องกัน data leakage และบันทึก/
   โหลดค่าที่ fit แล้วด้วย joblib เพื่อให้ฝั่งให้บริการใช้ค่าเดียวกับฝั่งเทรนเป๊ะ ๆ
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ชื่อคอลัมน์ฟีเจอร์ที่ผ่าน PCA มาแล้วในชุดข้อมูลต้นทาง (V1 ถึง V28)
V_COLUMNS = [f"V{i}" for i in range(1, 29)]

# คอลัมน์ดิบที่โมดูลนี้ต้องการเป็นอย่างน้อยเพื่อทำงานได้
REQUIRED_RAW_COLUMNS = ["Time", "Amount"] + V_COLUMNS

SECONDS_PER_HOUR = 3600
HOURS_PER_DAY = 24

RecordLike = Union[pd.DataFrame, dict, list]


def _to_dataframe(X: RecordLike) -> pd.DataFrame:
    """แปลง input ให้เป็น pandas.DataFrame เสมอ ไม่ว่าจะถูกเรียกจากฝั่งเทรน
    (ส่ง DataFrame เป็นก้อน) หรือฝั่งให้บริการ (ส่ง dict 1 รายการที่ได้จากการ
    parse JSON ของ request) นี่คือจุดเดียวในโค้ดที่ตัดสินว่า "หนึ่งแถวของข้อมูล"
    หน้าตาเป็นอย่างไร ทำให้ทั้งสองฝั่งไม่มีทางแปลงข้อมูลเพี้ยนไปจากกัน
    """
    if isinstance(X, pd.DataFrame):
        return X.copy()
    if isinstance(X, dict):
        return pd.DataFrame([X])
    if isinstance(X, list):
        return pd.DataFrame(X)
    raise TypeError(
        f"ไม่รองรับชนิดข้อมูล {type(X).__name__} — ต้องเป็น pandas.DataFrame, "
        "dict (1 รายการ) หรือ list ของ dict เท่านั้น"
    )


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "ข้อมูลขาดคอลัมน์ที่จำเป็นสำหรับแปลงฟีเจอร์: "
            f"{missing} (ต้องมีอย่างน้อย {REQUIRED_RAW_COLUMNS})"
        )
    if (df["Amount"] < 0).any():
        raise ValueError(
            "พบ Amount ติดลบ — log1p ใช้กับค่าติดลบไม่ได้ตามหลักสถิติ "
            "ควรถูกดักไว้ตั้งแต่ชั้น schema validation (validate.py) แล้ว"
        )


def encode_time_cyclical(df: pd.DataFrame) -> pd.DataFrame:
    """แปลงคอลัมน์ ``Time`` (จำนวนวินาทีนับต่อเนื่องจากธุรกรรมแรกในชุดข้อมูล)
    ให้เป็นชั่วโมงของวัน (0–23) แล้วเข้ารหัสแบบวงกลม (cyclical encoding) ด้วย
    sine และ cosine

    ทำไมต้องเข้ารหัสแบบวงกลม (อธิบายสำหรับวันนำเสนอ):
        ถ้าใช้ hour_of_day เป็นตัวเลขธรรมดา (0, 1, 2, ..., 23) โมเดลจะ "เห็น"
        ว่า 23:00 กับ 00:00 ห่างกันถึง 23 หน่วย ทั้งที่ในความเป็นจริงเวลาทั้งสอง
        ห่างกันแค่ 1 ชั่วโมง (เที่ยงคืนของอีกวัน) การฉายชั่วโมงลงบนวงกลมหนึ่งหน่วย
        (unit circle) ด้วยคู่ค่า (sin, cos) ทำให้ระยะห่างทางเรขาคณิตระหว่างจุด
        สองจุดสะท้อนระยะห่างของเวลาจริงเสมอ 23:00 กับ 00:00 จึงกลายเป็นจุดที่
        อยู่ใกล้กันมากบนวงกลม (ระยะห่าง = 2*sin(pi/24) ซึ่งเท่ากับระยะห่างของ
        ชั่วโมงติดกันคู่อื่น ๆ ทุกคู่) และต้องใช้ "สอง" ค่า (sin และ cos) คู่กัน
        เพราะค่าใดค่าหนึ่งเพียงอย่างเดียวไม่สามารถระบุตำแหน่งบนวงกลมได้หนึ่งเดียว
        (เช่น sin ของชั่วโมงที่ 6 กับชั่วโมงที่ 18 มีค่าต่างกัน แต่ cos เท่ากัน
        ต้องดูคู่กันถึงจะแยกสองชั่วโมงนี้ออกจากกันได้)

    ข้อสมมติ/ข้อจำกัด: ชุดข้อมูลนี้ให้มาเฉพาะ "วินาทีที่ผ่านไปนับจากธุรกรรมแรก"
    ไม่ใช่เวลานาฬิกาจริง จึงตั้งสมมติฐานว่า Time = 0 ตรงกับเวลา 00:00 น. ถ้า
    ธุรกรรมแรกของชุดข้อมูลจริง ๆ ไม่ได้เกิดตอนเที่ยงคืน ชั่วโมงของวันที่คำนวณได้
    จะเลื่อนคลาดเคลื่อนไปเท่ากับเวลาที่ธุรกรรมแรกเกิดขึ้นจริง (เป็นข้อจำกัดที่ต้อง
    ระบุไว้ในรายงาน)
    """
    df = df.copy()
    hour_of_day = (df["Time"] // SECONDS_PER_HOUR) % HOURS_PER_DAY
    angle = 2 * np.pi * hour_of_day / HOURS_PER_DAY
    df["hour_of_day"] = hour_of_day.astype(int)
    df["hour_sin"] = np.sin(angle)
    df["hour_cos"] = np.cos(angle)
    return df


OUTPUT_COLUMNS = (
    ["hour_sin", "hour_cos", "amount_log_scaled"]
    + [f"{c}_scaled" for c in V_COLUMNS]
)


class FraudFeatureTransformer(BaseEstimator, TransformerMixin):
    """Transformer หลักของโปรเจค ใช้ interface แบบ scikit-learn (fit/transform)
    เพื่อให้ต่อเข้ากับ ``sklearn.pipeline.Pipeline`` ได้โดยตรง และบันทึก/โหลด
    ด้วย ``joblib`` ได้ในไฟล์เดียวพร้อมกับโมเดล

    ทำ 3 อย่าง:
      1. เข้ารหัสเวลาแบบวงกลม (sin/cos) — ดู :func:`encode_time_cyclical`
      2. แปลง ``Amount`` ด้วย ``log1p`` แล้ว scale ด้วย StandardScaler เพราะ
         Amount เบ้ขวาแรงมาก (รายการเล็กจำนวนมาก รายการใหญ่ไม่กี่รายการ)
         log1p บีบหางขวาที่ยาวให้สั้นลงก่อน แล้วจึง scale ให้มีค่าเฉลี่ย 0
         ส่วนเบี่ยงเบนมาตรฐาน 1 เพื่อให้อยู่ในสเกลใกล้เคียงกับฟีเจอร์อื่น ๆ
      3. Scale ``V1``–``V28`` ด้วยค่าเฉลี่ยและส่วนเบี่ยงเบนมาตรฐานที่ ``fit``
         จาก "ชุดฝึกเท่านั้น"

    คำเตือนสำคัญที่สุดของคลาสนี้: ``fit()`` ต้องถูกเรียกกับชุดฝึก (train split)
    เท่านั้น ห้ามเรียกกับ validation/test หรือข้อมูลทั้งก้อนก่อนแบ่ง เพราะค่า
    เฉลี่ย/ส่วนเบี่ยงเบนที่ได้จะ "เห็น" สถิติของข้อมูลที่ควรเป็นความลับ (test set)
    ทำให้ผลประเมินโมเดลดีเกินจริง (data leakage) ส่วน ``transform()`` เรียกได้
    กับข้อมูลชุดใดก็ได้ (train/valid/test/production) โดยใช้ค่าที่ fit ไว้แล้ว
    ตัวชี้วัดคุณภาพนี้ถูกพิสูจน์ด้วย unit test ใน ``tests/test_features.py``
    """

    def __init__(self) -> None:
        self.amount_scaler_: StandardScaler | None = None
        self.v_scaler_: StandardScaler | None = None

    def fit(self, X: RecordLike, y=None) -> "FraudFeatureTransformer":
        df = _to_dataframe(X)
        _validate_columns(df)

        log_amount = np.log1p(df["Amount"]).to_numpy().reshape(-1, 1)
        self.amount_scaler_ = StandardScaler().fit(log_amount)

        self.v_scaler_ = StandardScaler().fit(df[V_COLUMNS].to_numpy())

        self.n_features_in_ = len(REQUIRED_RAW_COLUMNS)
        return self

    def transform(self, X: RecordLike) -> pd.DataFrame:
        if self.amount_scaler_ is None or self.v_scaler_ is None:
            raise RuntimeError(
                "ยังไม่ได้ fit() — ต้องเรียก fit(train_df) ด้วยชุดฝึกก่อนเสมอ "
                "จึงจะ transform() ได้ (ดูคำอธิบายเรื่อง data leakage ใน docstring)"
            )

        df = _to_dataframe(X)
        _validate_columns(df)
        df = encode_time_cyclical(df)

        log_amount = np.log1p(df["Amount"]).to_numpy().reshape(-1, 1)
        amount_scaled = self.amount_scaler_.transform(log_amount).ravel()

        v_scaled = self.v_scaler_.transform(df[V_COLUMNS].to_numpy())
        v_scaled_df = pd.DataFrame(
            v_scaled,
            columns=[f"{c}_scaled" for c in V_COLUMNS],
            index=df.index,
        )

        out = pd.DataFrame(
            {
                "hour_sin": df["hour_sin"].to_numpy(),
                "hour_cos": df["hour_cos"].to_numpy(),
                "amount_log_scaled": amount_scaled,
            },
            index=df.index,
        )
        out = pd.concat([out, v_scaled_df], axis=1)

        # ลำดับคอลัมน์ผลลัพธ์ตายตัวเสมอ ไม่ขึ้นกับลำดับคอลัมน์ของ input
        return out[OUTPUT_COLUMNS]

    def fit_transform(self, X: RecordLike, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(OUTPUT_COLUMNS)


def save_transformer(transformer: FraudFeatureTransformer, path: Union[str, Path]) -> None:
    """บันทึก transformer ที่ fit แล้วลงไฟล์เดียวด้วย joblib เพื่อให้ฝั่งให้
    บริการ (serving) โหลดไปใช้ transform ข้อมูลใหม่ได้ทันทีโดยไม่ต้อง fit ซ้ำ
    และรับประกันว่าค่าเฉลี่ย/ส่วนเบี่ยงเบนที่ใช้ตอนให้บริการตรงกับตอนเทรนเป๊ะ ๆ
    """
    joblib.dump(transformer, Path(path))


def load_transformer(path: Union[str, Path]) -> FraudFeatureTransformer:
    """โหลด transformer ที่บันทึกไว้ด้วย :func:`save_transformer` กลับมาใช้งาน"""
    return joblib.load(Path(path))


TARGET_COLUMN = "Class"


def split_xy(df: pd.DataFrame):
    """แยกฟีเจอร์ (X) กับป้ายกำกับ (y) ออกจาก DataFrame"""
    return df.drop(columns=[TARGET_COLUMN]), df[TARGET_COLUMN]


def build_pipeline(estimator) -> Pipeline:
    """ต่อ FraudFeatureTransformer เข้ากับโมเดลเป็น Pipeline เดียว"""
    return Pipeline([("features", FraudFeatureTransformer()), ("model", estimator)])