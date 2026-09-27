"""Ingest, validate, split, train, gate, register, and promote a model."""

import io
import json
import os
import time

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from prefect import flow, task
from sklearn.metrics import recall_score

from fraud.bootstrap import ensure_raw_data
from fraud.config import ROOT, load_params
from fraud.data import file_hash, load_raw, time_split
from fraud.evaluate import pr_auc
from fraud.features import split_xy
from fraud.gate import passes_gate
from fraud.train import train_baseline
from fraud.validate import validate


@task
def ingest():
    path = ensure_raw_data(ROOT / load_params()["data"]["raw_path"])
    return load_raw(path), file_hash(path)


@task
def check(frame):
    return validate(frame)


@task
def split(frame, data_version):
    train, validation, test = time_split(frame)
    output = ROOT / load_params()["data"]["processed_dir"]
    output.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("validation", validation), ("test", test)):
        part.to_csv(output / f"{name}.csv", index=False)
    metadata = {
        "raw_sha256": data_version,
        "rows": {"train": len(train), "validation": len(validation), "test": len(test)},
        "time_ranges": {
            name: [float(part["Time"].min()), float(part["Time"].max())]
            for name, part in (("train", train), ("validation", validation), ("test", test))
        },
    }
    (output / "data_version.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return train, validation, test


@task
def train(train_frame, validation_frame, data_version):
    return train_baseline(train_frame, validation_frame, data_version)


def measure_model(model, x_test, y_test):
    probabilities = model.predict_proba(x_test)[:, 1]
    elapsed = []
    for index in range(min(30, len(x_test))):
        start = time.perf_counter()
        model.predict_proba(x_test.iloc[[index]])
        elapsed.append((time.perf_counter() - start) * 1000)
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    return {
        "recall": float(recall_score(y_test, probabilities >= 0.5)),
        "pr_auc": pr_auc(y_test, probabilities),
        "p95_ms": float(np.percentile(elapsed, 95)),
        "model_mb": len(buffer.getvalue()) / 1_000_000,
    }


@task
def evaluate_gate_register(run_id, test_frame, data_version):
    params = load_params()
    name = params["registry"]["model_name"]
    champion_alias = params["registry"]["champion_alias"]
    challenger_alias = params["registry"]["challenger_alias"]
    x_test, y_test = split_xy(test_frame)
    candidate_uri = f"runs:/{run_id}/model"
    candidate = measure_model(mlflow.sklearn.load_model(candidate_uri), x_test, y_test)
    client = MlflowClient()
    champion = None
    try:
        client.get_model_version_by_alias(name, champion_alias)
    except MlflowException as exc:
        missing = exc.error_code in {"RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE"}
        if not missing or "not found" not in str(exc).lower():
            raise
        # A new registry or a model without this alias has no champion yet.
    else:
        champion = measure_model(
            mlflow.sklearn.load_model(f"models:/{name}@{champion_alias}"), x_test, y_test
        )
    passed, reasons = passes_gate(candidate, champion)
    evidence = {
        "status": "pass" if passed else "fail",
        "run_id": run_id,
        "data_version": data_version,
        "test_rows": len(test_frame),
        "candidate": candidate,
        "champion": champion,
        "gate": params["gate"],
        "reasons": reasons,
    }
    report = ROOT / "reports" / "pipeline-gate.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2, ensure_ascii=False))
    if not passed:
        raise ValueError("Model gate failed: " + "; ".join(reasons))
    version = mlflow.register_model(candidate_uri, name, await_registration_for=60)
    for key, value in candidate.items():
        client.set_model_version_tag(name, version.version, key, str(value))
    client.set_model_version_tag(name, version.version, "data_version", data_version)
    client.set_registered_model_alias(name, challenger_alias, version.version)
    client.set_registered_model_alias(name, champion_alias, version.version)
    return version.version


@flow(name="fraud-training-pipeline")
def training_pipeline():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", (ROOT / "mlruns").as_uri()))
    frame, data_version = ingest()
    checked = check(frame)
    train_frame, validation_frame, test_frame = split(checked, data_version)
    run_id = train(train_frame, validation_frame, data_version)
    return evaluate_gate_register(run_id, test_frame, data_version)


if __name__ == "__main__":
    training_pipeline()
