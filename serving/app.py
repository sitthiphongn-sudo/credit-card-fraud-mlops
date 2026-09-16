"""บริการทำนายแบบ real-time + cascade — ผู้รับผิดชอบ: สหรัฐ

/predict  validate → hard rules → pipeline (transform + model) → approve / review / decline
/health   สถานะและเวอร์ชันโมเดล
/metrics  สำหรับ Prometheus
TODO(สหรัฐ): โหลดจาก MLflow Registry (models:/fraud-detector@champion), เพิ่ม hard rules, threshold จาก gate
"""
import os
import time

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel

from fraud.validate import FEATURES, validate

MODEL_URI = os.getenv("MODEL_URI", "models:/fraud-detector@champion")
REQUESTS = Counter("fraud_requests_total", "requests", ["status"])
DECISIONS = Counter("fraud_decisions_total", "decisions", ["decision"])
LATENCY = Histogram("fraud_request_latency_seconds", "latency",
                    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0))

app = FastAPI(title="Fraud Detection API")
model = None


class Txn(BaseModel):
    instances: list[dict]


@app.on_event("startup")
def load_model():
    global model
    try:
        import mlflow
        model = mlflow.sklearn.load_model(MODEL_URI)
    except Exception as exc:  # ยังไม่มีโมเดลในทะเบียน
        print(f"model not loaded: {exc}")


@app.get("/health")
def health():
    return {"status": "ok" if model is not None else "degraded", "model_uri": MODEL_URI}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict(req: Txn):
    start = time.perf_counter()
    try:
        if model is None:
            raise HTTPException(503, "model not loaded")
        df = validate(pd.DataFrame(req.instances), with_target=False)[FEATURES]
        proba = model.predict_proba(df)[:, 1]
        decisions = ["decline" if p >= 0.8 else "review" if p >= 0.3 else "approve" for p in proba]
        for d in decisions:
            DECISIONS.labels(d).inc()
        REQUESTS.labels("200").inc()
        return {"probabilities": [round(float(p), 4) for p in proba], "decisions": decisions}
    except HTTPException:
        REQUESTS.labels("503").inc()
        raise
    except Exception as exc:
        REQUESTS.labels("400").inc()
        raise HTTPException(400, str(exc)) from exc
    finally:
        LATENCY.observe(time.perf_counter() - start)
