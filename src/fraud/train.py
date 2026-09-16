"""เทรนและบันทึกการทดลองลง MLflow — ผู้รับผิดชอบ: สิทธิพงษ์

บันทึกครบ 6 อย่าง: code version (git sha), data version (hash), params, metrics, artifacts, environment
TODO(สิทธิพงษ์): เพิ่ม run ของ LightGBM / XGBoost / การจัดการ imbalance
"""
import subprocess

import mlflow
from sklearn.linear_model import LogisticRegression

from fraud.config import load_params
from fraud.evaluate import pr_auc, recall_at_precision
from fraud.features import build_pipeline, split_xy


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def train_baseline(train_df, val_df, data_version: str):
    p = load_params()
    X_tr, y_tr = split_xy(train_df)
    X_va, y_va = split_xy(val_df)
    params = {"C": 1.0, "class_weight": "balanced", "max_iter": 1000}
    with mlflow.start_run(run_name="baseline-logreg") as run:
        mlflow.set_tags({"git_sha": git_sha(), "data_version": data_version})
        mlflow.log_params(params)
        pipe = build_pipeline(LogisticRegression(random_state=p["seed"], **params)).fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_va)[:, 1]
        mlflow.log_metrics({"val_pr_auc": pr_auc(y_va, proba),
                            "val_recall_at_p80": recall_at_precision(y_va, proba)})
        mlflow.sklearn.log_model(pipe, artifact_path="model", input_example=X_va.head(3))
        mlflow.log_artifact("requirements.txt")
        return run.info.run_id
