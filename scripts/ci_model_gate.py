"""Train a small smoke model, save metrics, and exit nonzero when the gate fails."""

import argparse
import hashlib
import io
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score

from fraud.config import load_params
from fraud.data import time_split
from fraud.evaluate import pr_auc
from fraud.features import build_pipeline, split_xy
from fraud.gate import passes_gate
from fraud.validate import validate


def run(sample: Path, output: Path) -> bool:
    frame = validate(pd.read_csv(sample))
    train, _, test = time_split(frame)
    x_train, y_train = split_xy(train)
    x_test, y_test = split_xy(test)
    model = build_pipeline(
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=load_params()["seed"])
    ).fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    elapsed = []
    for index in range(min(30, len(x_test))):
        start = time.perf_counter()
        model.predict_proba(x_test.iloc[[index]])
        elapsed.append((time.perf_counter() - start) * 1000)
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    candidate = {
        "recall": float(recall_score(y_test, probabilities >= 0.5)),
        "pr_auc": pr_auc(y_test, probabilities),
        "p95_ms": float(np.percentile(elapsed, 95)),
        "model_mb": len(buffer.getvalue()) / 1_000_000,
    }
    passed, reasons = passes_gate(candidate, None)
    evidence = {
        "status": "pass" if passed else "fail",
        "metrics": candidate,
        "gate": load_params()["gate"],
        "reasons": reasons,
        "sample_sha256": hashlib.sha256(sample.read_bytes()).hexdigest(),
        "test_rows": len(test),
        "test_fraud_rows": int(y_test.sum()),
        "sample_kind": "deterministic synthetic CI smoke sample",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2, ensure_ascii=False))
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, default=Path("data/sample/good.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/ci-model-gate.json"))
    args = parser.parse_args()
    raise SystemExit(0 if run(args.sample, args.output) else 1)
