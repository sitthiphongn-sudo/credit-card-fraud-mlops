"""Prefect DAG ทั้งกระบวนการ — ผู้รับผิดชอบ: ธรรมรักษ์

รัน: make all  (หรือ python pipelines/flow.py)
TODO(ธรรมรักษ์): evaluate บน test → gate → register → ตั้ง alias
"""
from prefect import flow, task

from fraud.config import ROOT, load_params
from fraud.data import file_hash, load_raw, time_split
from fraud.gate import (
    load_champion_metrics,
    model_size_mb,
    passes_gate,
    promote_to_champion,
    register_challenger,
)
from fraud.train import EXPERIMENTS_BY_NAME, run_experiment
from fraud.validate import validate_raw


@task
def ingest():
    p = load_params()
    path = ROOT / p["data"]["raw_path"]
    return load_raw(path), file_hash(path)


@task
def check(df):
    return validate_raw(df)

@task
def split(df):
    return time_split(df)


@task
@task
def train(
    train_df,
    val_df,
    test_df,
    data_version,
) -> tuple[str, dict]:
    result, _ = run_experiment(
        exp=EXPERIMENTS_BY_NAME["xgboost_class_weight"],
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        data_version=data_version,
    )

    run_id = result["run_id"]
    if not run_id:
        raise RuntimeError("training did not create an MLflow run")

    metrics = result["metrics"]
    candidate_metrics = {
        "recall": metrics["test_recall"],
        "pr_auc": metrics["test_pr_auc"],
        "p95_ms": metrics["latency_p95_ms"],
        "model_mb": metrics["model_mb"],
    }

    return run_id, candidate_metrics

@task
def add_model_size(
    run_id: str,
    candidate_metrics: dict,
) -> dict:
    """เติมขนาด artifact เมื่อผลจากการเทรนยังไม่มี model_mb."""
    complete_metrics = dict(candidate_metrics)

    if "model_mb" not in complete_metrics:
        complete_metrics["model_mb"] = model_size_mb(run_id)

    return complete_metrics

@task
def register_candidate(run_id: str, candidate_metrics: dict) -> str:
    """ลงทะเบียนโมเดลใหม่เป็น challenger."""
    return register_challenger(run_id, candidate_metrics)


@task
def promote_if_approved(
    version: str,
    candidate_metrics: dict,
    champion_metrics: dict | None = None,
) -> str:
    """ตรวจ gate และเลื่อนเป็น champion เฉพาะเมื่อผ่าน."""
    current_champion = champion_metrics

    if current_champion is None:
        current_champion = load_champion_metrics()

    passed, reasons = passes_gate(
        candidate_metrics,
        current_champion,
    )

    if not passed:
        details = "; ".join(reasons)
        raise RuntimeError(f"model gate failed: {details}")

    promote_to_champion(version)
    return version


@flow(name="fraud-model-release")
def release_pipeline(
    run_id: str,
    candidate_metrics: dict,
    champion_metrics: dict | None = None,
) -> str:
    complete_metrics = add_model_size(
        run_id,
        candidate_metrics,
    )
    version = register_candidate(
        run_id,
        complete_metrics,
    )

    return promote_if_approved(
        version,
        complete_metrics,
        champion_metrics,
    )

@flow(name="fraud-training-pipeline")
def training_pipeline():
    df, version = ingest()
    df = check(df)
    tr, va, te = split(df)

    run_id, candidate_metrics = train(
        tr,
        va,
        te,
        version,
    )

    return release_pipeline(
        run_id,
        candidate_metrics,
    )


if __name__ == "__main__":
    training_pipeline()