"""Prefect DAG ทั้งกระบวนการ — ผู้รับผิดชอบ: ธรรมรักษ์

รัน: make all  (หรือ python pipelines/flow.py)
TODO(ธรรมรักษ์): evaluate บน test → gate → register → ตั้ง alias
"""
from prefect import flow, task

from fraud.config import ROOT, load_params
from fraud.data import file_hash, load_raw, time_split
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


@flow(name="fraud-training-pipeline")
def training_pipeline():
    df, version = ingest()
    df = check(df)
    tr, va, te = split(df)
    return train(tr, va, version)


if __name__ == "__main__":
    training_pipeline()
