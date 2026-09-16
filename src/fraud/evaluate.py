"""ตัวชี้วัด — ผู้รับผิดชอบ: สิทธิพงษ์"""
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def pr_auc(y, proba) -> float:
    return float(average_precision_score(y, proba))


def recall_at_precision(y, proba, min_precision: float = 0.80) -> float:
    p, r, _ = precision_recall_curve(y, proba)
    ok = r[p >= min_precision]
    return float(ok.max()) if ok.size else 0.0


def best_cost_threshold(y, proba, amount, review_fee: float) -> tuple[float, float]:
    """เลือก threshold ที่ต้นทุนรวมต่ำสุด: FN เสีย Amount, FP เสียค่าตรวจสอบ"""
    y, proba, amount = map(np.asarray, (y, proba, amount))
    best = (0.5, np.inf)
    for t in np.linspace(0.01, 0.99, 99):
        pred = proba >= t
        cost = amount[(y == 1) & ~pred].sum() + review_fee * ((y == 0) & pred).sum()
        if cost < best[1]:
            best = (float(t), float(cost))
    return best
