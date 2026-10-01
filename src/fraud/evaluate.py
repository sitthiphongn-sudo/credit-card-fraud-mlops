"""ตัวชี้วัดของโมเดลตรวจจับธุรกรรมทุจริต — ผู้รับผิดชอบ: สิทธิพงษ์

ทำไมไม่ใช้ accuracy:
    fraud มีแค่ประมาณ 0.17% ของธุรกรรมทั้งหมด โมเดลที่ทายว่า "ไม่โกง" ทุกรายการจะได้
    accuracy ~99.8% ทั้งที่จับ fraud ไม่ได้เลยสักรายการ จึงใช้ตัวชี้วัดที่โฟกัสคลาส fraud แทน

ตัวชี้วัดที่ใช้ในโปรเจคนี้:
    - PR-AUC (average precision)  : optimizing metric หลัก เหมาะกับข้อมูลไม่สมดุล
    - Recall ที่ Precision >= 0.80  : จับ fraud ได้กี่ % เมื่อยอมให้แจ้งเตือนผิดไม่เกิน 20%
    - ต้นทุนทางธุรกิจ              : FN เสียเท่ากับยอดเงินของรายการที่หลุด (Amount)
                                     FP เสียค่าตรวจสอบคงที่ต่อรายการ (configs/params.yaml)
                                     ทั้งสองค่าต้องเป็นสกุลเงินเดียวกัน (ยูโร ตามหน่วยของ Amount)
    - ช่วงความเชื่อมั่นแบบ bootstrap : ชุดทดสอบมี fraud ไม่ถึงร้อยรายการ ตัวเลขเดี่ยวจึงแกว่งง่าย

กติกาสำคัญ: เลือก threshold จากชุด validation เท่านั้น แล้วค่อยนำ threshold นั้นไปวัดผลบนชุด test
ถ้าเลือก threshold บน test จะได้ผลดีเกินจริง (เท่ากับแอบดูข้อสอบ)
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

DEFAULT_THRESHOLDS = np.linspace(0.01, 0.99, 99)


def pr_auc(y, proba) -> float:
    """พื้นที่ใต้กราฟ Precision-Recall (average precision) — ยิ่งสูงยิ่งดี สูงสุด 1.0"""
    return float(average_precision_score(y, proba))


def recall_at_precision(y, proba, min_precision: float = 0.80) -> float:
    """Recall สูงสุดที่ทำได้ โดยที่ precision ยังไม่ต่ำกว่า ``min_precision``"""
    p, r, _ = precision_recall_curve(y, proba)
    ok = r[p >= min_precision]
    return float(ok.max()) if ok.size else 0.0


def confusion_at_threshold(y, proba, threshold: float) -> dict[str, int]:
    """นับ TP / FP / FN / TN เมื่อตัดสินว่า "โกง" ที่ proba >= threshold"""
    y = np.asarray(y).astype(int)
    pred = np.asarray(proba) >= threshold
    return {
        "tp": int(((y == 1) & pred).sum()),
        "fp": int(((y == 0) & pred).sum()),
        "fn": int(((y == 1) & ~pred).sum()),
        "tn": int(((y == 0) & ~pred).sum()),
    }


def cost_at_threshold(y, proba, amount, review_fee: float, threshold: float) -> dict[str, float]:
    """ต้นทุนรวมเมื่อใช้ threshold นี้

    - fn_cost : ยอดเงินของรายการโกงที่ปล่อยผ่าน (ธนาคารเสียเงินเท่ายอดนั้น)
    - fp_cost : จำนวนรายการปกติที่โดนส่งตรวจ x ค่าตรวจสอบต่อรายการ
    """
    y = np.asarray(y).astype(int)
    amount = np.asarray(amount, dtype=float)
    pred = np.asarray(proba) >= threshold
    fn_cost = float(amount[(y == 1) & ~pred].sum())
    fp_cost = float(review_fee * ((y == 0) & pred).sum())
    return {"fn_cost": fn_cost, "fp_cost": fp_cost, "total_cost": fn_cost + fp_cost}


def candidate_thresholds(proba, n_quantiles: int = 1000) -> np.ndarray:
    """ค่า threshold ที่จะลองไล่: ตาราง 0.01–0.99 รวมกับค่า quantile ของคะแนนจริง

    ต้องมีค่า quantile ด้วย เพราะโมเดลที่ถ่วงน้ำหนักคลาสมักให้คะแนนอัดกันอยู่แถว 0.99 ขึ้นไป
    ถ้าไล่แค่ 0.01–0.99 จะหา threshold ที่ดีที่สุดจริงไม่เจอ
    """
    quantiles = np.quantile(np.asarray(proba, dtype=float), np.linspace(0, 1, n_quantiles + 1))
    return np.unique(np.concatenate([DEFAULT_THRESHOLDS, quantiles]))


def best_cost_threshold(y, proba, amount, review_fee: float, thresholds=None) -> tuple[float, float]:
    """ไล่ threshold ทีละค่า แล้วเลือกค่าที่ต้นทุนรวมต่ำสุด คืนค่า (threshold, ต้นทุนรวม)

    ถ้าไม่ระบุ ``thresholds`` จะใช้ :func:`candidate_thresholds` (เรียงจากน้อยไปมาก)
    ถ้าต้นทุนเท่ากันหลายค่า จะเลือก threshold ที่ต่ำที่สุด (ค่าแรกที่เจอ) เพื่อให้จับ fraud ได้มากไว้ก่อน
    """
    candidates = candidate_thresholds(proba) if thresholds is None else np.asarray(thresholds, dtype=float)
    best = (0.5, np.inf)
    for t in candidates:
        cost = cost_at_threshold(y, proba, amount, review_fee, float(t))["total_cost"]
        if cost < best[1]:
            best = (float(t), float(cost))
    return best


def no_model_cost(y, amount) -> float:
    """ต้นทุนเมื่อไม่มีระบบตรวจจับเลย = ทุกรายการโกงหลุดผ่านหมด ใช้เป็นจุดตั้งต้นคำนวณเงินที่ประหยัดได้"""
    y = np.asarray(y).astype(int)
    return float(np.asarray(amount, dtype=float)[y == 1].sum())


def bootstrap_ci(
    y,
    proba,
    metric: Callable = pr_auc,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """ช่วงความเชื่อมั่น (1 - alpha) ของตัวชี้วัด ด้วยการสุ่มแบบคืนที่ (bootstrap)

    สุ่มแยกชั้น (stratified) คือสุ่ม fraud จากกลุ่ม fraud และสุ่มรายการปกติจากกลุ่มปกติ
    เพื่อให้ทุกรอบมี fraud จำนวนเท่าเดิม ไม่เกิดรอบที่ไม่มี fraud เลยจนคำนวณ PR-AUC ไม่ได้
    """
    y = np.asarray(y).astype(int)
    proba = np.asarray(proba, dtype=float)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if pos.size == 0 or neg.size == 0:
        raise ValueError("bootstrap ต้องมีทั้ง fraud และรายการปกติอย่างน้อยกลุ่มละ 1 รายการ")

    rng = np.random.default_rng(seed)
    scores = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([rng.choice(pos, pos.size), rng.choice(neg, neg.size)])
        scores[i] = metric(y[idx], proba[idx])
    low, high = np.quantile(scores, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def evaluate_scores(
    y,
    proba,
    amount,
    threshold: float,
    review_fee: float,
    min_precision: float = 0.80,
) -> dict[str, float]:
    """รวมตัวชี้วัดทั้งหมดของชุดข้อมูลหนึ่งชุด ที่ threshold ที่กำหนด (ใช้ threshold จากชุด validation)"""
    y = np.asarray(y).astype(int)
    c = confusion_at_threshold(y, proba, threshold)
    precision = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0
    recall = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
    cost = cost_at_threshold(y, proba, amount, review_fee, threshold)
    baseline = no_model_cost(y, amount)
    return {
        "pr_auc": pr_auc(y, proba),
        "roc_auc": float(roc_auc_score(y, proba)) if 0 < y.sum() < y.size else float("nan"),
        "recall_at_p80": recall_at_precision(y, proba, min_precision),
        "precision": float(precision),
        "recall": float(recall),
        **{k: float(v) for k, v in c.items()},
        **cost,
        "no_model_cost": baseline,
        "savings": baseline - cost["total_cost"],
    }
