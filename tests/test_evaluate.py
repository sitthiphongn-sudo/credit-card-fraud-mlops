"""เทสต์ตัวชี้วัดใน src/fraud/evaluate.py — ผู้รับผิดชอบ: สิทธิพงษ์"""

import numpy as np
import pytest

from fraud.evaluate import (
    best_cost_threshold,
    bootstrap_ci,
    confusion_at_threshold,
    cost_at_threshold,
    evaluate_scores,
    no_model_cost,
    pr_auc,
    recall_at_precision,
)

# ชุดข้อมูลของเล่น 6 รายการ: fraud 2 รายการ (index 4, 5)
Y = np.array([0, 0, 0, 0, 1, 1])
PROBA = np.array([0.05, 0.10, 0.40, 0.70, 0.60, 0.95])
AMOUNT = np.array([10.0, 20.0, 30.0, 40.0, 500.0, 1000.0])
FEE = 50.0


def test_perfect_ranking_gives_pr_auc_of_one():
    assert pr_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)


def test_recall_at_precision_is_zero_when_precision_never_reaches_target():
    # อันดับคะแนนกลับด้านทั้งหมด precision ไม่มีทางถึง 0.8
    assert recall_at_precision([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9], 0.8) == 0.0


def test_confusion_counts_match_manual_count():
    # threshold 0.5 → ทายว่าโกง index 3, 4, 5
    assert confusion_at_threshold(Y, PROBA, 0.5) == {"tp": 2, "fp": 1, "fn": 0, "tn": 3}


def test_cost_uses_amount_for_missed_fraud_and_fixed_fee_for_false_alarm():
    # threshold 0.8 → จับได้แค่ index 5 ปล่อย index 4 (500 บาท) หลุด และไม่มี FP
    cost = cost_at_threshold(Y, PROBA, AMOUNT, FEE, 0.8)
    assert cost == {"fn_cost": 500.0, "fp_cost": 0.0, "total_cost": 500.0}
    # threshold 0.5 → จับได้ครบ แต่แจ้งเตือนผิด 1 รายการ เสียค่าตรวจสอบ 50
    assert cost_at_threshold(Y, PROBA, AMOUNT, FEE, 0.5)["total_cost"] == 50.0


def test_best_cost_threshold_prefers_catching_expensive_fraud():
    # ยอมแจ้งเตือนผิด 1 รายการ (เสีย 50) ดีกว่าปล่อย fraud 500 บาทหลุด
    # threshold ที่ได้ต้องอยู่ช่วง 0.41–0.60 ซึ่งจับ fraud ได้ครบทั้งสองรายการ
    threshold, cost = best_cost_threshold(Y, PROBA, AMOUNT, FEE)
    assert cost == 50.0
    assert 0.40 < threshold <= 0.60


def test_min_precision_rules_out_cheap_but_noisy_thresholds():
    """ถ้าดูแค่ต้นทุน จะได้ threshold 0.41–0.60 (ต้นทุน 50) แต่ precision แค่ 2/3
    บังคับ precision >= 0.8 → ต้องขยับขึ้นไปช่วง 0.71–0.95 จับได้แค่รายการ 1000 ยอมให้ 500 หลุด"""
    threshold, cost = best_cost_threshold(Y, PROBA, AMOUNT, FEE, min_precision=0.8)
    assert cost == 500.0
    assert 0.70 < threshold <= 0.95


def test_min_precision_falls_back_to_most_precise_threshold_when_unreachable():
    """fraud ได้คะแนนต่ำกว่ารายการปกติทุกตัว precision สูงสุดที่ทำได้คือ 1/3 (แจ้งทุกรายการ) ต้องไม่ล่ม"""
    y, proba, amount = [1, 0, 0], [0.2, 0.5, 0.9], [100.0, 1.0, 1.0]
    threshold, cost = best_cost_threshold(y, proba, amount, 50.0, min_precision=0.8)
    assert threshold <= 0.2
    assert cost == 100.0


def test_best_cost_threshold_accepts_custom_candidates():
    threshold, cost = best_cost_threshold(Y, PROBA, AMOUNT, FEE, thresholds=[0.5, 0.8])
    assert (threshold, cost) == (0.5, 50.0)


def test_no_model_cost_is_total_fraud_amount():
    assert no_model_cost(Y, AMOUNT) == 1500.0


def test_evaluate_scores_reports_savings_against_no_model():
    m = evaluate_scores(Y, PROBA, AMOUNT, threshold=0.5, review_fee=FEE)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(1.0)
    assert m["savings"] == pytest.approx(1500.0 - 50.0)
    for key in ("pr_auc", "roc_auc", "recall_at_p80", "tp", "fp", "fn", "tn", "total_cost"):
        assert key in m


def test_bootstrap_ci_is_reproducible_and_contains_point_estimate():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(500, dtype=int), np.ones(20, dtype=int)]
    proba = np.clip(rng.normal(0.2, 0.1, y.size) + 0.4 * y, 0, 1)

    first = bootstrap_ci(y, proba, n_boot=300, seed=7)
    second = bootstrap_ci(y, proba, n_boot=300, seed=7)
    assert first == second
    low, high = first
    assert low <= pr_auc(y, proba) <= high


def test_bootstrap_ci_rejects_data_without_fraud():
    with pytest.raises(ValueError):
        bootstrap_ci([0, 0, 0], [0.1, 0.2, 0.3], n_boot=10)


def test_threshold_search_can_go_above_0_99_for_overconfident_models():
    """โมเดลถ่วงน้ำหนักคลาสให้คะแนนรายการปกติสูงถึง 0.995 ถ้าไล่แค่ 0.01–0.99 จะหา threshold ที่ถูกไม่เจอ"""
    y = np.array([0, 0, 0, 1])
    proba = np.array([0.990, 0.992, 0.995, 0.999])
    amount = np.array([10.0, 10.0, 10.0, 1000.0])
    threshold, cost = best_cost_threshold(y, proba, amount, review_fee=50.0)
    assert cost == 0.0
    assert 0.995 < threshold <= 0.999
