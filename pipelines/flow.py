"""Prefect DAG ทั้งกระบวนการ — ผู้รับผิดชอบ: ธรรมรักษ์

รัน: make all  (หรือ python pipelines/flow.py)
TODO(ธรรมรักษ์): evaluate บน test → gate → register → ตั้ง alias
"""
from prefect import flow, task

from fraud.config import ROOT, load_params
from fraud.data import file_hash, load_raw, time_split
from fraud.gate import (
    passes_gate,
    promote_to_champion,
    register_challenger,
)
from fraud.train import train_baseline
from fraud.validate import validate


@task
def ingest():
    p = load_params()
    path = ROOT / p["data"]["raw_path"]
    return load_raw(path), file_hash(path)


@task
def check(df):
    return validate(df)  # ข้อมูลเสีย → raise → flow หยุด


@task
def split(df):
    return time_split(df)


@task
def train(train_df, val_df, data_version):
    return train_baseline(train_df, val_df, data_version)

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
    passed, reasons = passes_gate(
        candidate_metrics,
        champion_metrics,
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
    version = register_candidate(run_id, candidate_metrics)

    return promote_if_approved(
        version,
        candidate_metrics,
        champion_metrics,
    )

@flow(name="fraud-training-pipeline")
def training_pipeline():
    df, version = ingest()
    df = check(df)
    tr, va, te = split(df)
    return train(tr, va, version)


if __name__ == "__main__":
    training_pipeline()
