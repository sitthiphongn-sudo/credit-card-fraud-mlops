"""โหลดข้อมูลดิบ, สร้าง data version (hash) และแบ่งชุดตามเวลา — ผู้รับผิดชอบ: เปรมสิริวัฒณ์"""
import hashlib
from pathlib import Path

import pandas as pd

from fraud.config import ROOT, load_params


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    p = load_params()
    return pd.read_csv(path or ROOT / p["data"]["raw_path"])


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """แบ่งตามลำดับเวลา ป้องกันข้อมูลอนาคตรั่วเข้าชุดฝึก ผลเหมือนเดิมทุกครั้ง"""
    s = load_params()["data"]["split"]
    df = df.sort_values("Time", kind="mergesort").reset_index(drop=True)
    n = len(df)
    a, b = int(n * s["train"]), int(n * (s["train"] + s["val"]))
    return df.iloc[:a], df.iloc[a:b], df.iloc[b:]
