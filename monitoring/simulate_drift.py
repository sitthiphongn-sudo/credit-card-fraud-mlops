"""จำลอง Data Drift และ Concept Drift บนชุดทดสอบ — ผู้รับผิดชอบ: โชติกานต์

data drift   : Amount สูงขึ้น, การกระจายของ V14/V17 เลื่อน (P(X) เปลี่ยน)
concept drift: X เหมือนเดิม แต่ label เปลี่ยน (P(y|X) เปลี่ยน)
"""
import numpy as np
import pandas as pd


def data_drift(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    out = df.copy()
    rng = np.random.default_rng(seed)
    out["Amount"] = out["Amount"] * 1.8
    for c in ["V14", "V17"]:
        out[c] = out[c] + rng.normal(1.5, 0.3, len(out))
    return out


def concept_drift(df: pd.DataFrame, seed: int = 42, share: float = 0.05) -> pd.DataFrame:
    out = df.copy()
    rng = np.random.default_rng(seed)
    small = (out["Amount"] < 5) & (out["Class"] == 0)
    flip = small & (rng.random(len(out)) < share)
    out.loc[flip, "Class"] = 1
    return out
