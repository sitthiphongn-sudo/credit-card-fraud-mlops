"""Prefect DAG ทั้งกระบวนการ — ผู้รับผิดชอบ: ธรรมรักษ์

รัน: make all  (หรือ python pipelines/flow.py)
ขั้นตอน: ingest → check → split → train → register challenger → gate → champion
"""
import argparse

from prefect import flow, task

from fraud.config import ROOT, load_params
from fraud.data import file_hash, load_raw, time_split
from fraud.gate import (
    candidate_from_best_model,
    load_champion_metrics,
    model_size_mb,
    passes_gate,
    promote_to_champion,
    register_challenger,
)
from fraud.train import EXPERIMENTS, run_all, select_best, write_reports
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
def train(
    train_df,
    val_df,
    test_df,
    data_version,
) -> tuple[str, dict]:
    """รันการทดลอง เลือกโมเดลจาก validation และส่ง test metrics เข้า Gate."""
    results = run_all(
        experiments=EXPERIMENTS,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        data_version=data_version,
        n_boot=200,
    )

    best = select_best(results)
    write_reports(results, best)

    if best is None:
        raise RuntimeError("training did not produce an eligible model")

    run_id = best["run_id"]
    if not run_id:
        raise RuntimeError("training did not create an MLflow run")

    candidate_metrics = candidate_from_best_model(best)

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


@flow(name="fraud-retrain-pipeline")
def retrain_pipeline(signal: str) -> str:
    """เริ่มวงจรเทรนใหม่เมื่อได้รับสัญญาณ RETRAIN เท่านั้น."""
    if signal.strip().upper() != "RETRAIN":
        return "ignored"

    return training_pipeline()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="รัน training pipeline หรือรับสัญญาณ retrain",
    )
    parser.add_argument(
        "--signal",
        choices=("TRAIN", "RETRAIN"),
        default="TRAIN",
        type=str.upper,
    )
    args = parser.parse_args()

    if args.signal == "RETRAIN":
        retrain_pipeline(args.signal)
    else:
        training_pipeline()


if __name__ == "__main__":
    main()
