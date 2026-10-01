"""เทสต์การเทรนและการทดลองใน src/fraud/train.py — ผู้รับผิดชอบ: สิทธิพงษ์

ใช้ข้อมูลสังเคราะห์ขนาดเล็ก (หน้าตาเหมือนชุดข้อมูล ULB) เพื่อให้รันใน CI ได้เร็วและไม่ต้องโหลดข้อมูลจริง
เทสต์ที่แตะ MLflow ใช้ MLflow ปลอม เพื่อตรวจว่าบันทึกครบ 6 อย่างโดยไม่ต้องเปิด tracking server
"""

import json
import sys
import types
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest

from fraud import train as T
from fraud.config import load_params
from fraud.features import V_COLUMNS

# ---------------------------------------------------------------------------
# ข้อมูลสังเคราะห์
# ---------------------------------------------------------------------------


def make_split(n: int, fraud_rate: float, seed: int, time_offset: float = 0.0) -> pd.DataFrame:
    """fraud แยกได้จาก V14 และ V17 (เหมือนข้อมูลจริง) ส่วน Amount ไม่มีสัญญาณเลย
    ดังนั้นโมเดล ML ควรชนะ baseline แบบกฎที่ดูแค่ยอดเงิน
    """
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < fraud_rate).astype(int)
    df = pd.DataFrame(rng.normal(size=(n, len(V_COLUMNS))), columns=V_COLUMNS)
    df["V14"] -= 3.0 * y
    df["V17"] -= 2.0 * y
    df.insert(0, "Time", time_offset + np.sort(rng.uniform(0, 50_000, n)))
    df["Amount"] = rng.gamma(2.0, 40.0, n).round(2)
    df["Class"] = y
    return df


@pytest.fixture(scope="module")
def splits():
    train = make_split(3000, 0.03, seed=1)
    val = make_split(1000, 0.03, seed=2, time_offset=50_000)
    test = make_split(1000, 0.03, seed=3, time_offset=100_000)
    return train, val, test


# ---------------------------------------------------------------------------
# MLflow ปลอม
# ---------------------------------------------------------------------------


class FakeMlflow(types.ModuleType):
    def __init__(self):
        super().__init__("mlflow")
        self.calls = {"tags": {}, "params": {}, "metrics": {}, "models": 0, "dicts": {}, "artifacts": []}
        self.experiment = None
        self.sklearn = types.SimpleNamespace(log_model=self._log_model)

    def set_experiment(self, name):
        self.experiment = name

    @contextmanager
    def start_run(self, run_name=None):
        self.calls["run_name"] = run_name
        yield types.SimpleNamespace(info=types.SimpleNamespace(run_id="fake-run-123"))

    def set_tags(self, tags):
        self.calls["tags"].update(tags)

    def log_params(self, params):
        self.calls["params"].update(params)

    def log_metrics(self, metrics):
        self.calls["metrics"].update(metrics)

    def log_dict(self, data, path):
        self.calls["dicts"][path] = data

    def log_artifact(self, path):
        self.calls["artifacts"].append(path)

    def _log_model(self, model, artifact_path, input_example=None):
        assert artifact_path == "model"
        assert input_example is not None and len(input_example) == 3
        self.calls["models"] += 1


@pytest.fixture
def fake_mlflow(monkeypatch):
    fake = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    return fake


# ---------------------------------------------------------------------------
# Baseline แบบกฎ
# ---------------------------------------------------------------------------


def test_rule_model_flags_exactly_the_amounts_above_the_cutoff(splits):
    train, val, _ = splits
    rule = T.AmountRuleModel(quantile=0.9).fit(train)
    expected = (val["Amount"] >= rule.cutoff_).astype(int).to_numpy()
    assert np.array_equal(rule.predict(val), expected)


def test_rule_model_score_increases_with_amount(splits):
    train, _, _ = splits
    rule = T.AmountRuleModel().fit(train)
    scores = rule.predict_proba(pd.DataFrame({"Amount": [1.0, 50.0, 500.0, 5000.0]}))[:, 1]
    assert np.all(np.diff(scores) >= 0)
    assert scores[-1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# การจัดการข้อมูลไม่สมดุล
# ---------------------------------------------------------------------------


def test_undersample_keeps_every_fraud_and_limits_normals(splits):
    train, _, _ = splits
    n_fraud = int(train["Class"].sum())
    out = T.undersample(train, ratio=10, seed=42)
    assert int(out["Class"].sum()) == n_fraud
    assert int((out["Class"] == 0).sum()) == 10 * n_fraud


def test_undersample_is_reproducible(splits):
    train, _, _ = splits
    assert T.undersample(train, 10, seed=1).equals(T.undersample(train, 10, seed=1))


def test_logreg_class_weight_switches_on_only_for_class_weight_strategy():
    _, weighted = T.build_estimator(T.EXPERIMENTS_BY_NAME["logreg_class_weight"], seed=42, pos_weight=30.0)
    _, plain = T.build_estimator(T.EXPERIMENTS_BY_NAME["logreg_none"], seed=42, pos_weight=30.0)
    assert weighted["class_weight"] == "balanced"
    assert plain["class_weight"] is None


def test_boosting_weight_strategies():
    assert T.boosting_weight("class_weight", 576.0) == 576.0
    assert T.boosting_weight("sqrt_weight", 576.0) == 24.0
    assert T.boosting_weight("none", 576.0) == 1.0


def test_review_fee_is_converted_to_euro_to_match_amount():
    """Amount ของ ULB เป็นยูโร แต่ค่าตรวจสอบตั้งเป็นบาท ต้องแปลงก่อนรวมต้นทุน ไม่งั้นหน่วยเงินปนกัน"""
    assert T.review_fee_eur({"review_fee_thb": 50.0, "eur_to_thb": 40.0}) == pytest.approx(1.25)


def test_lightgbm_params_guard_against_exploding_leaf_values():
    """LightGBM พังเมื่อ min_child_weight ต่ำ (ค่าเริ่มต้น 0.001) — ต้องตั้งไว้อย่างน้อย 1 เสมอ"""
    for name in ("lightgbm_class_weight", "lightgbm_none", "lightgbm_sqrt_weight"):
        assert T.EXPERIMENTS_BY_NAME[name].params["min_child_weight"] >= 1.0
        assert T.EXPERIMENTS_BY_NAME[name].params["reg_lambda"] > 0


def test_lightgbm_sqrt_weight_uses_square_root_of_training_ratio():
    pytest.importorskip("lightgbm")
    _, params = T.build_estimator(T.EXPERIMENTS_BY_NAME["lightgbm_sqrt_weight"], seed=42, pos_weight=576.0)
    assert params["scale_pos_weight"] == 24.0


def test_lightgbm_uses_scale_pos_weight_from_training_ratio():
    pytest.importorskip("lightgbm")
    _, params = T.build_estimator(T.EXPERIMENTS_BY_NAME["lightgbm_class_weight"], seed=42, pos_weight=577.0)
    assert params["scale_pos_weight"] == 577.0


# ---------------------------------------------------------------------------
# การทดลองหนึ่งรอบ
# ---------------------------------------------------------------------------


def test_run_experiment_reports_val_test_latency_and_size(splits):
    train, val, test = splits
    exp = T.EXPERIMENTS_BY_NAME["logreg_class_weight"]
    result, _ = T.run_experiment(exp, train, val, test, "abc123", n_boot=50, log_to_mlflow=False)
    m = result["metrics"]
    for key in (
        "val_pr_auc",
        "test_pr_auc",
        "test_pr_auc_ci_low",
        "test_pr_auc_ci_high",
        "test_recall_at_p80",
        "test_savings",
        "latency_p50_ms",
        "latency_p95_ms",
        "model_mb",
    ):
        assert key in m
    assert m["test_pr_auc_ci_low"] <= m["test_pr_auc"] <= m["test_pr_auc_ci_high"]
    assert 0.0 < result["threshold"] < 1.0
    assert result["data_version"] == "abc123"
    assert result["run_id"] is None
    cost = load_params()["cost"]
    assert result["review_fee_eur"] == pytest.approx(cost["review_fee_thb"] / cost["eur_to_thb"])


def test_threshold_is_chosen_from_validation_not_test(splits):
    """ถ้าแก้ป้ายกำกับของชุด test แล้ว threshold เปลี่ยน แปลว่ามีการแอบดูข้อสอบ — ต้องไม่เปลี่ยน"""
    train, val, test = splits
    shuffled = test.copy()
    shuffled["Class"] = shuffled["Class"].sample(frac=1.0, random_state=0).to_numpy()
    exp = T.EXPERIMENTS_BY_NAME["logreg_class_weight"]
    a, _ = T.run_experiment(exp, train, val, test, n_boot=0, log_to_mlflow=False)
    b, _ = T.run_experiment(exp, train, val, shuffled, n_boot=0, log_to_mlflow=False)
    assert a["threshold"] == b["threshold"]


def test_ml_model_beats_amount_rule_when_signal_is_not_in_amount(splits):
    train, val, test = splits
    rule, _ = T.run_experiment(T.EXPERIMENTS_BY_NAME["rule_amount_p995"], train, val, test, n_boot=0,
                               log_to_mlflow=False)
    logreg, _ = T.run_experiment(T.EXPERIMENTS_BY_NAME["logreg_class_weight"], train, val, test, n_boot=0,
                                 log_to_mlflow=False)
    assert logreg["metrics"]["test_pr_auc"] > rule["metrics"]["test_pr_auc"]


def test_trained_pipeline_accepts_a_single_raw_json_record(splits):
    """โมเดลที่ได้ต้องรับข้อมูลดิบทีละรายการได้ เหมือนตอนที่ API ส่งเข้ามา"""
    train, val, _ = splits
    _, model = T.run_experiment(T.EXPERIMENTS_BY_NAME["logreg_class_weight"], train, val, n_boot=0,
                                log_to_mlflow=False)
    record = val.drop(columns=["Class"]).iloc[0].to_dict()
    proba = model.predict_proba(pd.DataFrame([record]))[:, 1]
    assert proba.shape == (1,) and 0.0 <= proba[0] <= 1.0


# ---------------------------------------------------------------------------
# การเลือกโมเดลและรายงาน
# ---------------------------------------------------------------------------


def _fake_result(name, family, val_pr_auc, p95=5.0, mb=1.0):
    return {
        "name": name,
        "family": family,
        "imbalance": "class_weight",
        "threshold": 0.5,
        "run_id": f"run-{name}",
        "data_version": "abc",
        "metrics": {"val_pr_auc": val_pr_auc, "latency_p95_ms": p95, "model_mb": mb},
    }


def test_select_best_ignores_rule_and_models_that_break_the_gate():
    results = [
        _fake_result("rule", "rule", 0.99),
        _fake_result("too_slow", "lightgbm", 0.95, p95=500.0),
        _fake_result("too_big", "xgboost", 0.94, mb=500.0),
        _fake_result("ok_low", "logreg", 0.70),
        _fake_result("ok_high", "lightgbm", 0.80),
    ]
    assert T.select_best(results)["name"] == "ok_high"


def test_select_best_returns_none_when_nothing_qualifies():
    assert T.select_best([_fake_result("rule", "rule", 0.9)]) is None


def test_write_reports_creates_table_and_handoff_file(splits, tmp_path):
    train, val, test = splits
    results = [
        T.run_experiment(T.EXPERIMENTS_BY_NAME[name], train, val, test, "abc", n_boot=20,
                         log_to_mlflow=False)[0]
        for name in ("rule_amount_p995", "logreg_class_weight")
    ]
    best = T.select_best(results)
    T.write_reports(results, best, out_dir=tmp_path)

    assert (tmp_path / "experiments.csv").exists()
    markdown = (tmp_path / "experiments.md").read_text(encoding="utf-8")
    assert "logreg_class_weight **(เลือก)**" in markdown
    assert "ประหยัดได้ test (บาท)" in markdown
    handoff = json.loads((tmp_path / "best_model.json").read_text(encoding="utf-8"))
    assert handoff["name"] == "logreg_class_weight"
    assert 0.0 < handoff["threshold"] < 1.0


# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------


def test_log_run_records_all_six_items(splits, fake_mlflow):
    train, val, test = splits
    exp = T.EXPERIMENTS_BY_NAME["logreg_class_weight"]
    result, _ = T.run_experiment(exp, train, val, test, "abc123", n_boot=20, log_to_mlflow=True)
    calls = fake_mlflow.calls
    assert result["run_id"] == "fake-run-123"
    assert fake_mlflow.experiment == T.EXPERIMENT_NAME
    assert "git_sha" in calls["tags"]  # 1) เวอร์ชันโค้ด
    assert calls["tags"]["data_version"] == "abc123"  # 2) เวอร์ชันข้อมูล
    assert calls["params"]["model__class_weight"] == "balanced"  # 3) ไฮเปอร์พารามิเตอร์
    assert "threshold" in calls["params"]
    assert calls["params"]["currency"] == "EUR"
    assert "test_pr_auc" in calls["metrics"]  # 4) ตัวชี้วัด
    assert all(np.isfinite(v) for v in calls["metrics"].values())
    assert calls["models"] == 1 and "result.json" in calls["dicts"]  # 5) ไฟล์ผลลัพธ์
    assert "python_version" in calls["tags"]  # 6) สภาพแวดล้อม


def test_train_baseline_keeps_the_signature_used_by_pipeline_flow(splits, fake_mlflow):
    """pipelines/flow.py เรียก train_baseline(train, val, data_version) แล้วรอ run_id กลับมา"""
    train, val, _ = splits
    assert T.train_baseline(train, val, "abc123") == "fake-run-123"
