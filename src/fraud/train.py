"""เทรนโมเดล ทดลองเปรียบเทียบ และบันทึกลง MLflow — ผู้รับผิดชอบ: สิทธิพงษ์

รันทั้งหมด:        python -m fraud.train
รันบางการทดลอง:   python -m fraud.train --only logreg_class_weight,lightgbm_class_weight
ไม่บันทึก MLflow:  python -m fraud.train --no-mlflow

ลำดับการทำงานของแต่ละการทดลอง (run_experiment):
    1. จัดการข้อมูลไม่สมดุลเฉพาะ "ชุดฝึก" ตามกลยุทธ์ของการทดลองนั้น (none / class_weight / undersample)
    2. เทรนบนชุดฝึก
    3. เลือก threshold ที่ต้นทุนต่ำสุดจาก "ชุด validation" (ห้ามเลือกบนชุด test)
    4. วัดผลบนชุด test ด้วย threshold นั้น + ช่วงความเชื่อมั่นของ PR-AUC แบบ bootstrap
    5. วัดเวลาทำนายทีละรายการ (p50 / p95) และขนาดไฟล์โมเดล เพื่อเทียบกับเกณฑ์ gate
    6. บันทึกลง MLflow ครบ 6 อย่าง: เวอร์ชันโค้ด (git sha), เวอร์ชันข้อมูล (hash),
       ไฮเปอร์พารามิเตอร์, ตัวชี้วัด, ไฟล์ผลลัพธ์ (โมเดล + result.json), และสภาพแวดล้อม
       (requirements ที่ MLflow บันทึกพร้อมโมเดล + requirements.txt ของโปรเจค + เวอร์ชัน Python)

ขอบเขต: ไฟล์นี้ "ไม่" ขึ้นทะเบียนโมเดลและไม่ตั้ง alias — เป็นหน้าที่ของ gate/pipeline (ธรรมรักษ์)
ผลที่ส่งต่อให้เพื่อนอยู่ใน reports/experiments/best_model.json (run_id + threshold + ตัวชี้วัด)
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from fraud.config import ROOT, load_params
from fraud.evaluate import best_cost_threshold, bootstrap_ci, evaluate_scores, pr_auc
from fraud.features import TARGET_COLUMN, build_pipeline, split_xy

log = logging.getLogger("fraud.train")

EXPERIMENT_NAME = "fraud-detection"
REPORT_DIR = ROOT / "reports" / "experiments"
UNDERSAMPLE_RATIO = 10  # เก็บรายการปกติไว้ 10 รายการต่อ fraud 1 รายการ
LATENCY_SAMPLES = 200  # จำนวนรายการที่ใช้วัดเวลาทำนายทีละรายการ


# ---------------------------------------------------------------------------
# Baseline แบบกฎ
# ---------------------------------------------------------------------------


class AmountRuleModel(BaseEstimator, ClassifierMixin):
    """Baseline แบบกฎ: ยอดเงินตั้งแต่เปอร์เซ็นไทล์ที่ ``quantile`` ของชุดฝึกขึ้นไป ถือว่าน่าสงสัย

    ``predict_proba`` คืนค่า "สัดส่วนของรายการในชุดฝึกที่ยอดเงินไม่เกินรายการนี้" (empirical CDF)
    ค่านี้เรียงลำดับตาม Amount เหมือนกฎทุกประการ จึงวัด PR-AUC ได้ด้วยตัวชี้วัดชุดเดียวกับโมเดล ML
    และถ้าตัดที่ ``quantile`` จะได้ผลเท่ากับกฎ "Amount >= cutoff" พอดี

    มีไว้เพื่อพิสูจน์ด้วยตัวเลขว่า ML ดีกว่ากฎธรรมดาจริง ตามเงื่อนไขของโจทย์
    """

    def __init__(self, quantile: float = 0.995) -> None:
        self.quantile = quantile

    def fit(self, X: pd.DataFrame, y=None) -> AmountRuleModel:
        self.train_amounts_ = np.sort(np.asarray(X["Amount"], dtype=float))
        self.cutoff_ = float(np.quantile(self.train_amounts_, self.quantile))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        amount = np.asarray(X["Amount"], dtype=float)
        score = np.searchsorted(self.train_amounts_, amount, side="right") / self.train_amounts_.size
        return np.column_stack([1.0 - score, score])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (np.asarray(X["Amount"], dtype=float) >= self.cutoff_).astype(int)


# ---------------------------------------------------------------------------
# นิยามการทดลอง
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Experiment:
    """หนึ่งการทดลอง = ตระกูลโมเดล + วิธีจัดการข้อมูลไม่สมดุล + ไฮเปอร์พารามิเตอร์"""

    name: str
    family: str  # rule | logreg | lightgbm | xgboost
    imbalance: str  # none | class_weight | sqrt_weight | undersample
    params: dict = field(default_factory=dict)


LOGREG_PARAMS = {"C": 1.0, "max_iter": 1000}
LIGHTGBM_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    # สองค่านี้คือการแก้ LightGBM ที่พัง (ดูคำอธิบายที่ EXPERIMENTS ด้านล่าง)
    "min_child_weight": 1.0,  # ค่าเริ่มต้นของ LightGBM คือ 0.001 ต่ำเกินไปสำหรับข้อมูลที่ fraud มี 0.2%
    "reg_lambda": 1.0,  # L2 ช่วยหดค่าของใบให้ไม่สุดโต่ง
}
XGBOOST_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

EXPERIMENTS = [
    Experiment("rule_amount_p995", "rule", "none", {"quantile": 0.995}),
    # เปรียบเทียบวิธีจัดการข้อมูลไม่สมดุลบนโมเดลเดียวกัน (Logistic Regression)
    Experiment("logreg_none", "logreg", "none", LOGREG_PARAMS),
    Experiment("logreg_class_weight", "logreg", "class_weight", LOGREG_PARAMS),
    Experiment("logreg_undersample", "logreg", "undersample", LOGREG_PARAMS),
    # โมเดลต้นไม้แบบ boosting ถ่วงน้ำหนักคลาสด้วย scale_pos_weight
    Experiment("lightgbm_class_weight", "lightgbm", "class_weight", LIGHTGBM_PARAMS),
    # ประวัติ: รอบแรก (28 ก.ย.) LightGBM ทุกแบบพัง val PR-AUC 0.007–0.08 แม้ไม่ถ่วงน้ำหนัก
    # สาเหตุ: min_child_weight เริ่มต้นของ LightGBM = 0.001 ทำให้มีใบที่ผลรวม hessian เกือบศูนย์
    # ค่าของใบ = ผลรวม gradient / ผลรวม hessian จึงพุ่งเป็นหลักหมื่น (|raw score| สูงสุด ~35,000)
    # โมเดลลู่ออกตั้งแต่ตอนเทรน (train PR-AUC แค่ 0.33) ไม่ใช่ overfit
    # แก้ด้วย min_child_weight=1 (เท่ากับค่าเริ่มต้นของ XGBoost) + L2 → val PR-AUC 0.787
    Experiment("lightgbm_none", "lightgbm", "none", LIGHTGBM_PARAMS),
    Experiment("lightgbm_sqrt_weight", "lightgbm", "sqrt_weight", LIGHTGBM_PARAMS),
    Experiment("xgboost_class_weight", "xgboost", "class_weight", XGBOOST_PARAMS),
]
EXPERIMENTS_BY_NAME = {e.name: e for e in EXPERIMENTS}


def undersample(train_df: pd.DataFrame, ratio: int, seed: int) -> pd.DataFrame:
    """สุ่มลดรายการปกติให้เหลือ ``ratio`` เท่าของ fraud เก็บ fraud ไว้ครบทุกรายการ

    ใช้กับชุดฝึกเท่านั้น ห้ามใช้กับ validation/test เพราะจะทำให้ผลประเมินไม่สะท้อนสัดส่วนจริง
    """
    y = train_df[TARGET_COLUMN]
    fraud = train_df[y == 1]
    normal = train_df[y == 0]
    keep = min(len(normal), ratio * len(fraud))
    sampled = normal.sample(n=keep, random_state=seed)
    return pd.concat([fraud, sampled]).sort_index()


def boosting_weight(imbalance: str, pos_weight: float) -> float:
    """ค่า scale_pos_weight ของโมเดล boosting ตามกลยุทธ์

    - class_weight : จำนวนรายการปกติ / จำนวน fraud (~497 ในชุดฝึกจริง) ถ่วงเต็มที่
    - sqrt_weight  : รากที่สองของค่าข้างบน (~22) ถ่วงแบบนุ่มลง กันคะแนนอิ่มตัวจน PR-AUC พัง
    - อื่น ๆ        : 1.0 ไม่ถ่วง ปล่อยให้การเลือก threshold จากต้นทุนจัดการความไม่สมดุลแทน
    """
    if imbalance == "class_weight":
        return pos_weight
    if imbalance == "sqrt_weight":
        return float(np.sqrt(pos_weight))
    return 1.0


def build_estimator(exp: Experiment, seed: int, pos_weight: float):
    """สร้างโมเดลของการทดลอง คืน (โมเดล, พารามิเตอร์ที่ใช้จริงทั้งหมด สำหรับบันทึก MLflow)

    โมเดล ML ทุกตัวถูกห่อด้วย ``build_pipeline`` ของนาคินทร์ เพื่อให้การแปลงข้อมูลติดไปกับโมเดล
    ฝั่งให้บริการจึงส่งข้อมูลดิบเข้ามาได้เลย ไม่เกิด training-serving skew
    """
    weighted = exp.imbalance == "class_weight"

    if exp.family == "rule":
        params = dict(exp.params)
        return AmountRuleModel(**params), params

    if exp.family == "logreg":
        params = {**exp.params, "class_weight": "balanced" if weighted else None, "random_state": seed}
        return build_pipeline(LogisticRegression(**params)), params

    if exp.family == "lightgbm":
        from lightgbm import LGBMClassifier

        params = {
            **exp.params,
            "scale_pos_weight": boosting_weight(exp.imbalance, pos_weight),
            "random_state": seed,
            "n_jobs": -1,
            "verbose": -1,
        }
        return build_pipeline(LGBMClassifier(**params)), params

    if exp.family == "xgboost":
        from xgboost import XGBClassifier

        params = {
            **exp.params,
            "scale_pos_weight": boosting_weight(exp.imbalance, pos_weight),
            "tree_method": "hist",
            "eval_metric": "aucpr",
            "random_state": seed,
            "n_jobs": -1,
        }
        return build_pipeline(XGBClassifier(**params)), params

    raise ValueError(f"ไม่รู้จักตระกูลโมเดล: {exp.family}")


# ---------------------------------------------------------------------------
# การวัดผลที่ไม่ใช่ความแม่น: เวลาและขนาด (ใช้เทียบกับเกณฑ์ gate ใน configs/params.yaml)
# ---------------------------------------------------------------------------


def review_fee_eur(cost: dict) -> float:
    """ค่าตรวจสอบต่อรายการในหน่วยยูโร ให้หน่วยตรงกับ Amount ของชุดข้อมูล ULB

    สมมติฐานค่าตรวจสอบตั้งไว้เป็นบาทใน configs/params.yaml จึงต้องหารด้วยอัตราแลกเปลี่ยนก่อน
    ถ้าไม่แปลง จะเอาบาทไปบวกกับยูโรตรง ๆ ทำให้ต้นทุน false positive แพงเกินจริงราว 38 เท่า
    และ threshold ที่ได้จะสูงเกินไป (แจ้งเตือนน้อยเกินไป)
    """
    return cost["review_fee_thb"] / cost["eur_to_thb"]


def measure_latency(model, X: pd.DataFrame, n: int = LATENCY_SAMPLES) -> tuple[float, float]:
    """วัดเวลาทำนายทีละ 1 รายการ (แบบเดียวกับตอนให้บริการจริง) คืน (p50, p95) หน่วยมิลลิวินาที"""
    rows = X.head(n)
    model.predict_proba(rows.iloc[[0]])  # warm-up รอบแรกไม่นับ
    times = []
    for i in range(len(rows)):
        start = time.perf_counter()
        model.predict_proba(rows.iloc[[i]])
        times.append((time.perf_counter() - start) * 1000)
    p50, p95 = np.percentile(times, [50, 95])
    return float(p50), float(p95)


def model_size_mb(model) -> float:
    """ขนาดไฟล์โมเดลเมื่อบันทึกด้วย joblib (MB)"""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.joblib"
        joblib.dump(model, path)
        return path.stat().st_size / (1024 * 1024)


# ---------------------------------------------------------------------------
# เวอร์ชันโค้ด
# ---------------------------------------------------------------------------


def git_sha() -> str:
    try:
        cmd = ["git", "rev-parse", "--short", "HEAD"]
        return subprocess.check_output(cmd, text=True, cwd=ROOT, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    """True ถ้ามีไฟล์แก้ไขที่ยังไม่ commit — run นั้นจะอ้างอิงเวอร์ชันโค้ดได้ไม่แม่นยำ"""
    try:
        cmd = ["git", "status", "--porcelain"]
        out = subprocess.check_output(cmd, text=True, cwd=ROOT, stderr=subprocess.DEVNULL)
        return bool(out.strip())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# รันหนึ่งการทดลอง
# ---------------------------------------------------------------------------


def run_experiment(
    exp: Experiment,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame | None = None,
    data_version: str = "unknown",
    n_boot: int = 1000,
    log_to_mlflow: bool = True,
) -> tuple[dict, object]:
    """เทรนและประเมินหนึ่งการทดลอง คืน (ผลลัพธ์แบบ dict, โมเดลที่เทรนแล้ว)"""
    p = load_params()
    seed = p["seed"]
    fee = review_fee_eur(p["cost"])  # ยูโร หน่วยเดียวกับ Amount

    fit_df = undersample(train_df, UNDERSAMPLE_RATIO, seed) if exp.imbalance == "undersample" else train_df
    X_fit, y_fit = split_xy(fit_df)
    n_fraud = int(y_fit.sum())
    if n_fraud == 0:
        raise ValueError("ชุดฝึกไม่มี fraud เลย เทรนโมเดลไม่ได้")
    pos_weight = float((y_fit == 0).sum() / n_fraud)

    model, used_params = build_estimator(exp, seed, pos_weight)
    start = time.perf_counter()
    model.fit(X_fit, y_fit)
    train_seconds = time.perf_counter() - start

    # เลือก threshold จาก validation เท่านั้น
    X_val, y_val = split_xy(val_df)
    val_proba = model.predict_proba(X_val)[:, 1]
    threshold, _ = best_cost_threshold(y_val, val_proba, X_val["Amount"], fee)
    val_metrics = evaluate_scores(y_val, val_proba, X_val["Amount"], threshold, fee)
    metrics = {f"val_{k}": v for k, v in val_metrics.items()}

    if test_df is not None:
        X_test, y_test = split_xy(test_df)
        test_proba = model.predict_proba(X_test)[:, 1]
        test_metrics = evaluate_scores(y_test, test_proba, X_test["Amount"], threshold, fee)
        metrics.update({f"test_{k}": v for k, v in test_metrics.items()})
        if n_boot > 0 and 0 < int(y_test.sum()) < len(y_test):
            low, high = bootstrap_ci(y_test, test_proba, pr_auc, n_boot=n_boot, seed=seed)
            metrics.update({"test_pr_auc_ci_low": low, "test_pr_auc_ci_high": high})

    p50, p95 = measure_latency(model, X_val)
    metrics.update(
        {
            "latency_p50_ms": p50,
            "latency_p95_ms": p95,
            "model_mb": model_size_mb(model),
            "train_seconds": train_seconds,
        }
    )

    result = {
        "name": exp.name,
        "family": exp.family,
        "imbalance": exp.imbalance,
        "threshold": threshold,
        "params": used_params,
        "train_rows": len(fit_df),
        "train_fraud": n_fraud,
        "data_version": data_version,
        "review_fee_eur": fee,
        "eur_to_thb": p["cost"]["eur_to_thb"],
        "metrics": metrics,
        "run_id": None,
    }
    if log_to_mlflow:
        result["run_id"] = log_run(result, model, X_val)
    log.info(
        "%-24s val PR-AUC %.4f | threshold %.4f | p95 %.1f ms",
        exp.name,
        metrics["val_pr_auc"],
        threshold,
        p95,
    )
    return result, model


def log_run(result: dict, model, X_example: pd.DataFrame) -> str:
    """บันทึกหนึ่งการทดลองลง MLflow ให้ครบ 6 อย่างตามเกณฑ์ คืน run_id

    ใช้ tracking server ตามตัวแปร MLFLOW_TRACKING_URI (ถ้าไม่ตั้งจะเก็บในโฟลเดอร์ ./mlruns)
    """
    import mlflow

    p = load_params()
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=result["name"]) as run:
        # 1) เวอร์ชันโค้ด  2) เวอร์ชันข้อมูล  6) สภาพแวดล้อม
        mlflow.set_tags(
            {
                "git_sha": git_sha(),
                "git_dirty": str(git_dirty()),
                "data_version": result["data_version"],
                "model_family": result["family"],
                "imbalance": result["imbalance"],
                "python_version": platform.python_version(),
                "sklearn_version": sklearn.__version__,
            }
        )
        # 3) ไฮเปอร์พารามิเตอร์
        mlflow.log_params(
            {
                **{f"model__{k}": v for k, v in result["params"].items()},
                "imbalance": result["imbalance"],
                "threshold": result["threshold"],
                "seed": p["seed"],
                "currency": p["cost"]["currency"],
                "review_fee_thb": p["cost"]["review_fee_thb"],
                "review_fee_eur": result["review_fee_eur"],
                "eur_to_thb": result["eur_to_thb"],
                "train_rows": result["train_rows"],
                "train_fraud": result["train_fraud"],
            }
        )
        # 4) ตัวชี้วัด (ข้ามค่า NaN เช่น ROC-AUC ของชุดที่ไม่มี fraud)
        mlflow.log_metrics({k: float(v) for k, v in result["metrics"].items() if np.isfinite(v)})
        # 5) ไฟล์ผลลัพธ์: โมเดล (พร้อม requirements ของสภาพแวดล้อมที่ MLflow สร้างให้) + สรุปผล
        # แปลงเป็น float ก่อน เพราะ Time ในไฟล์ดิบเป็นจำนวนเต็ม MLflow จะเตือนเรื่อง schema ของคอลัมน์ int
        example = X_example.head(3).astype(float)
        mlflow.sklearn.log_model(model, artifact_path="model", input_example=example)
        mlflow.log_dict(_jsonable(result), "result.json")
        for extra in (ROOT / "requirements.txt", ROOT / p["data"]["processed_dir"] / "data_version.json"):
            if extra.exists():
                mlflow.log_artifact(str(extra))
        return run.info.run_id


def _jsonable(result: dict) -> dict:
    return json.loads(json.dumps(result, default=str))


# ---------------------------------------------------------------------------
# รันทุกการทดลอง เลือกตัวที่ดีที่สุด และเขียนรายงาน
# ---------------------------------------------------------------------------


def run_all(
    experiments: list[Experiment],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    data_version: str,
    n_boot: int = 1000,
    log_to_mlflow: bool = True,
) -> list[dict]:
    """รันทุกการทดลอง ถ้าไลบรารีของโมเดลใดไม่ได้ติดตั้ง จะข้ามการทดลองนั้นพร้อมแจ้งเตือน"""
    results = []
    for exp in experiments:
        try:
            result, _ = run_experiment(exp, train_df, val_df, test_df, data_version, n_boot, log_to_mlflow)
        except ImportError as err:
            log.warning("ข้าม %s เพราะยังไม่ได้ติดตั้งไลบรารี: %s", exp.name, err)
            continue
        results.append(result)
    return results


def select_best(results: list[dict]) -> dict | None:
    """เลือกโมเดลสุดท้ายด้วยชุด validation (ไม่ใช้ test เพื่อไม่ให้ผล test ลำเอียง)

    เงื่อนไข: ต้องไม่ใช่ baseline แบบกฎ และต้องผ่านเกณฑ์เวลา p95 กับขนาดไฟล์ใน configs/params.yaml
    จากนั้นเลือกตัวที่ val PR-AUC สูงสุด ถ้าเท่ากันให้เลือกตัวที่เร็วกว่า
    """
    gate = load_params()["gate"]
    eligible = [
        r
        for r in results
        if r["family"] != "rule"
        and r["metrics"]["latency_p95_ms"] <= gate["max_p95_latency_ms"]
        and r["metrics"]["model_mb"] <= gate["max_model_mb"]
    ]
    if not eligible:
        log.warning("ไม่มีโมเดล ML ตัวใดผ่านเกณฑ์เวลาและขนาดใน configs/params.yaml")
        return None
    return max(eligible, key=lambda r: (r["metrics"]["val_pr_auc"], -r["metrics"]["latency_p95_ms"]))


REPORT_COLUMNS = [
    ("name", "การทดลอง"),
    ("imbalance", "จัดการ imbalance"),
    ("val_pr_auc", "val PR-AUC"),
    ("test_pr_auc", "test PR-AUC"),
    ("test_pr_auc_ci", "test PR-AUC 95% CI"),
    ("test_recall_at_p80", "test Recall@P≥0.8"),
    ("threshold", "threshold"),
    ("test_precision", "test Precision"),
    ("test_recall", "test Recall"),
    ("test_savings", "ประหยัดได้ test (EUR)"),
    ("test_savings_thb", "ประหยัดได้ test (บาท)"),
    ("latency_p95_ms", "p95 (ms)"),
    ("model_mb", "ขนาด (MB)"),
]


def results_table(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        m = r["metrics"]
        ci = (
            f"{m['test_pr_auc_ci_low']:.3f}–{m['test_pr_auc_ci_high']:.3f}"
            if "test_pr_auc_ci_low" in m
            else ""
        )
        rows.append(
            {
                "name": r["name"],
                "imbalance": r["imbalance"],
                "val_pr_auc": m["val_pr_auc"],
                "test_pr_auc": m.get("test_pr_auc", np.nan),
                "test_pr_auc_ci": ci,
                "test_recall_at_p80": m.get("test_recall_at_p80", np.nan),
                "threshold": r["threshold"],
                "test_precision": m.get("test_precision", np.nan),
                "test_recall": m.get("test_recall", np.nan),
                "test_savings": m.get("test_savings", np.nan),
                "test_savings_thb": m.get("test_savings", np.nan) * r.get("eur_to_thb", np.nan),
                "latency_p95_ms": m["latency_p95_ms"],
                "model_mb": m["model_mb"],
                "run_id": r["run_id"],
            }
        )
    return pd.DataFrame(rows)


def _fmt(value) -> str:
    if isinstance(value, float):
        if np.isnan(value):
            return "-"
        return f"{value:,.2f}" if abs(value) >= 100 else f"{value:.4f}"
    return str(value)


def write_reports(results: list[dict], best: dict | None, out_dir: Path = REPORT_DIR) -> Path:
    """เขียนตารางเปรียบเทียบ (csv + markdown) และไฟล์ส่งต่อ best_model.json (ถ้ามีโมเดลที่เลือกได้)"""
    out_dir.mkdir(parents=True, exist_ok=True)
    table = results_table(results)
    table.to_csv(out_dir / "experiments.csv", index=False)

    header = "| " + " | ".join(title for _, title in REPORT_COLUMNS) + " |"
    divider = "|" + "---|" * len(REPORT_COLUMNS)
    money = (
        f" · ต้นทุนคิดเป็นยูโร (หน่วยของ Amount) ค่าตรวจสอบ {results[0]['review_fee_eur']:.2f} EUR/รายการ"
        f" · แปลงเป็นบาทที่ {results[0]['eur_to_thb']} บาท/ยูโร"
        if results and "eur_to_thb" in results[0]
        else ""
    )
    lines = [
        "# ผลการทดลอง",
        "",
        "เลือก threshold จากชุด validation แล้ววัดผลบนชุด test · เงินที่ประหยัดได้เทียบกับกรณีไม่มีระบบ" + money,
        "",
        header,
        divider,
    ]
    for _, row in table.iterrows():
        cells = [_fmt(row[col]) for col, _ in REPORT_COLUMNS]
        mark = " **(เลือก)**" if best and row["name"] == best["name"] else ""
        cells[0] = f"{cells[0]}{mark}"
        lines.append("| " + " | ".join(cells) + " |")
    summary = (
        f"**โมเดลที่เลือก:** `{best['name']}` — val PR-AUC สูงสุดในกลุ่มที่ผ่านเกณฑ์เวลาและขนาด"
        if best
        else "**ยังเลือกโมเดลไม่ได้:** ไม่มีโมเดล ML ตัวใดผ่านเกณฑ์เวลาและขนาด"
    )
    lines += ["", summary, ""]
    md_path = out_dir / "experiments.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    if best is None:
        return md_path

    handoff = {
        "name": best["name"],
        "run_id": best["run_id"],
        "model_uri": f"runs:/{best['run_id']}/model" if best["run_id"] else None,
        "threshold": best["threshold"],
        "data_version": best["data_version"],
        "currency": "EUR",
        "review_fee_eur": best.get("review_fee_eur"),
        "eur_to_thb": best.get("eur_to_thb"),
        "metrics": best["metrics"],
    }
    path = out_dir / "best_model.json"
    path.write_text(json.dumps(handoff, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# โหลดข้อมูลที่แบ่งแล้ว (ของเปรมสิริวัฒณ์)
# ---------------------------------------------------------------------------


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """อ่านชุดข้อมูลที่แบ่งแล้วจาก data/processed/ ถ้ายังไม่มีจะเรียก prepare_data() ให้สร้างก่อน"""
    processed = ROOT / load_params()["data"]["processed_dir"]
    paths = [processed / f"{name}.csv" for name in ("train", "val", "test")]
    version_path = processed / "data_version.json"

    if all(path.exists() for path in paths) and version_path.exists():
        train, val, test = (pd.read_csv(path) for path in paths)
        meta = json.loads(version_path.read_text(encoding="utf-8"))
    else:
        from fraud.data import prepare_data

        train, val, test, meta = prepare_data()
    return train, val, test, meta.get("raw_hash", "unknown")


# ---------------------------------------------------------------------------
# ฟังก์ชันเดิมที่ pipelines/flow.py เรียกใช้ — คงชื่อและพารามิเตอร์ไว้ไม่ให้ flow ของเพื่อนพัง
# ---------------------------------------------------------------------------


def train_baseline(train_df: pd.DataFrame, val_df: pd.DataFrame, data_version: str) -> str | None:
    """เทรน Logistic Regression แบบถ่วงน้ำหนักคลาสหนึ่งตัว บันทึก MLflow แล้วคืน run_id"""
    result, _ = run_experiment(
        EXPERIMENTS_BY_NAME["logreg_class_weight"],
        train_df,
        val_df,
        test_df=None,
        data_version=data_version,
        n_boot=0,
    )
    return result["run_id"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="เทรนและเปรียบเทียบโมเดลตรวจจับ fraud")
    parser.add_argument("--only", help="ชื่อการทดลองคั่นด้วยจุลภาค เช่น logreg_class_weight,lightgbm_class_weight")
    parser.add_argument("--bootstrap", type=int, default=1000, help="จำนวนรอบ bootstrap (0 = ไม่ทำ)")
    parser.add_argument("--no-mlflow", action="store_true", help="ไม่บันทึกลง MLflow (ใช้ตอนทดสอบเร็ว ๆ)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    experiments = EXPERIMENTS
    if args.only:
        names = [n.strip() for n in args.only.split(",") if n.strip()]
        unknown = [n for n in names if n not in EXPERIMENTS_BY_NAME]
        if unknown:
            parser.error(f"ไม่รู้จักการทดลอง: {unknown} (ที่มี: {list(EXPERIMENTS_BY_NAME)})")
        experiments = [EXPERIMENTS_BY_NAME[n] for n in names]

    train, val, test, data_version = load_splits()
    results = run_all(experiments, train, val, test, data_version, args.bootstrap, not args.no_mlflow)
    best = select_best(results)
    path = write_reports(results, best)
    if best:
        log.info("โมเดลที่เลือก: %s (run_id=%s) · บันทึกผลส่งต่อที่ %s", best["name"], best["run_id"], path)


if __name__ == "__main__":
    main()
