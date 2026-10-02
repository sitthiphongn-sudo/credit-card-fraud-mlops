"""Fraud Detection API — ผู้รับผิดชอบ: สหรัฐ (Serving / FastAPI / Docker)

ข้อตกลงทีม (อัปงาน.pdf หัวข้อ 4 และ 7):
- โหลดโมเดลจาก MLflow Model Registry: models:/fraud-detector@champion (ห้ามใช้ models_cache/)
- threshold อ่านจาก param "threshold" ของ run ที่ champion ชี้อยู่ (rollback แล้ว threshold ย้อนตามเอง)
- ตัดสินด้วย predict_proba(...)[:, 1] >= threshold (ไม่ใช้ model.predict ที่ตัดที่ 0.5)
- /metrics มี 4 metric: fraud_requests_total{status}, fraud_request_latency_seconds,
  fraud_predictions_total{label}, fraud_score
- ข้อมูลผิดปกติทุกแบบที่ /predict ตอบ 400 และนับเป็น status="bad_request"
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn  # import ตรง ๆ (mlflow.sklearn เป็น lazy module)
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from mlflow.tracking import MlflowClient
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from prometheus_client import multiprocess as prometheus_multiprocess
from pydantic import BaseModel, Field

logger = logging.getLogger("serving")

MODEL_NAME = "fraud-detector"
MODEL_ALIAS = "champion"
MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
PREDICT_PATH = "/predict"

# MLflow server ใน docker compose อาจยังไม่พร้อมตอน API เริ่ม จึงลองโหลดซ้ำก่อนยอมแพ้
MODEL_LOAD_RETRIES = int(os.getenv("MODEL_LOAD_RETRIES", "30"))
MODEL_LOAD_RETRY_SECONDS = float(os.getenv("MODEL_LOAD_RETRY_SECONDS", "2"))

FEATURE_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
AMOUNT_INDEX = FEATURE_COLUMNS.index("Amount")


# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
# ใช้ registry แยก เพื่อให้ /metrics มีเฉพาะ metric ของระบบนี้
METRICS_REGISTRY = CollectorRegistry()

fraud_requests_total = Counter(
    "fraud_requests_total",
    "Total /predict requests by status (success, bad_request, error).",
    labelnames=["status"],
    registry=METRICS_REGISTRY,
)

fraud_request_latency_seconds = Histogram(
    "fraud_request_latency_seconds",
    "Latency of /predict requests in seconds.",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    registry=METRICS_REGISTRY,
)

fraud_predictions_total = Counter(
    "fraud_predictions_total",
    "Total fraud predictions by label (fraud, not_fraud).",
    labelnames=["label"],
    registry=METRICS_REGISTRY,
)

fraud_score = Histogram(
    "fraud_score",
    "Distribution of fraud probability scores.",
    buckets=(0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.51, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
    registry=METRICS_REGISTRY,
)

# ให้ series ของแต่ละ status/label ปรากฏใน /metrics ตั้งแต่เริ่ม (ค่า 0) เพื่อให้ alert/Grafana query ได้ทันที
for _status in ("success", "bad_request", "error"):
    fraud_requests_total.labels(status=_status)
for _label in ("fraud", "not_fraud"):
    fraud_predictions_total.labels(label=_label)


# ---------------------------------------------------------------------------
# Champion model state
# ---------------------------------------------------------------------------
model = None
threshold: float | None = None
model_version: str | None = None
model_run_id: str | None = None


class TransactionData(BaseModel):
    # รับค่าดิบก่อน (object) เพื่อให้ข้อมูลผิดประเภทถูกตรวจเองแล้วตอบ 400
    features: list[object] = Field(
        ...,
        description="30 numerical features in order: Time, V1..V28, Amount",
    )


def _threshold_from_run(client: MlflowClient, run_id: str) -> float:
    run = client.get_run(run_id)
    try:
        value = float(run.data.params["threshold"])
    except KeyError as exc:
        raise RuntimeError(f"Champion run {run_id} has no 'threshold' parameter") from exc
    except ValueError as exc:
        raise RuntimeError(f"Champion run {run_id} has a non-numeric threshold") from exc

    if not 0.0 <= value <= 1.0:
        raise RuntimeError(f"Champion threshold must be within [0, 1], got {value}")
    return value


def load_champion_model() -> None:
    """โหลด champion จาก MLflow Registry และอ่าน threshold จาก run เดียวกัน

    ตรวจ alias ก่อนและหลังโหลด ถ้า champion ถูกย้ายระหว่างโหลด (เช่นกำลัง rollback)
    จะโหลดใหม่ เพื่อให้ model, threshold และ model_version มาจาก version เดียวกันเสมอ
    """
    global model, threshold, model_version, model_run_id

    client = MlflowClient()

    for _ in range(3):
        mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        loaded_model = mlflow.sklearn.load_model(MODEL_URI)
        mv_after = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        if str(mv_after.version) == str(mv.version):
            break
    else:
        raise RuntimeError("Champion alias kept changing while loading the model")

    if not hasattr(loaded_model, "predict_proba"):
        raise RuntimeError("Champion model does not support predict_proba")

    loaded_threshold = _threshold_from_run(client, mv.run_id)

    model = loaded_model
    threshold = loaded_threshold
    model_version = str(mv.version)
    model_run_id = mv.run_id
    logger.info(
        "Loaded %s version=%s run_id=%s threshold=%s",
        MODEL_URI,
        model_version,
        model_run_id,
        threshold,
    )


def _load_with_retry() -> None:
    attempts = max(1, MODEL_LOAD_RETRIES)
    for attempt in range(1, attempts + 1):
        try:
            load_champion_model()
            return
        except Exception as exc:  # รายงานแล้วลองใหม่ ถ้าครบรอบจึงโยนต่อ
            if attempt == attempts:
                raise
            logger.warning(
                "Loading %s failed (attempt %d/%d): %s",
                MODEL_URI,
                attempt,
                attempts,
                exc,
            )
            time.sleep(MODEL_LOAD_RETRY_SECONDS)


def _is_ready() -> bool:
    return model is not None and threshold is not None and model_version is not None


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_with_retry()
    yield


app = FastAPI(
    title="Credit Card Fraud Detection API",
    version="1.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request metrics + error mapping
# ---------------------------------------------------------------------------
def _status_label(http_status: int) -> str:
    if http_status < 400:
        return "success"
    if http_status == 400:
        return "bad_request"
    return "error"


@app.middleware("http")
async def record_predict_metrics(request: Request, call_next):
    """นับทุก request ของ /predict (รวม JSON เสียที่ไม่ถึง handler) และจับ latency"""
    if request.url.path != PREDICT_PATH:
        return await call_next(request)

    started_at = time.perf_counter()
    http_status = 500
    try:
        response = await call_next(request)
        http_status = response.status_code
        return response
    finally:
        fraud_requests_total.labels(status=_status_label(http_status)).inc()
        fraud_request_latency_seconds.observe(time.perf_counter() - started_at)


@app.exception_handler(RequestValidationError)
async def request_validation_to_400(request: Request, exc: RequestValidationError):
    """JSON เสีย, ไม่มี key features หรือ features ไม่ใช่ list -> 400 ตามข้อตกลง (ไม่ใช่ 422)"""
    errors = [
        {"loc": list(err.get("loc", ())), "msg": err.get("msg", "")}
        for err in exc.errors()
    ]
    return JSONResponse(
        status_code=400,
        content={"detail": "Invalid request body", "errors": errors},
    )


def _validate_and_build_dataframe(features: list[object]) -> pd.DataFrame:
    """ตรวจข้อมูลแล้วสร้าง DataFrame 1 แถวสำหรับโมเดล (raise ValueError ถ้าข้อมูลผิด)"""
    if len(features) != len(FEATURE_COLUMNS):
        raise ValueError(f"Expected {len(FEATURE_COLUMNS)} features, got {len(features)}.")

    values: list[float] = []
    for index, value in enumerate(features):
        name = FEATURE_COLUMNS[index]
        # รับเฉพาะตัวเลข JSON จริง ๆ: string ("abc" หรือ "1.5"), bool, null, list, object ผิดหมด
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Feature '{name}' (index {index}) must be a number.")

        try:
            number = float(value)
        except OverflowError as exc:  # จำนวนเต็มใหญ่เกินช่วง float
            raise ValueError(f"Feature '{name}' (index {index}) is out of range.") from exc
        if not np.isfinite(number):
            raise ValueError(f"Feature '{name}' (index {index}) must be finite, got {value}.")
        values.append(number)

    if values[AMOUNT_INDEX] < 0:
        raise ValueError("Amount must be greater than or equal to 0.")

    return pd.DataFrame([values], columns=FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post(PREDICT_PATH)
def predict(data: TransactionData):
    if not _is_ready():
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    try:
        input_df = _validate_and_build_dataframe(data.features)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        fraud_probability = float(model.predict_proba(input_df)[0][1])
    except Exception as exc:
        # ข้อมูลผ่านการตรวจแล้ว ถ้ายังพังแปลว่าเป็นปัญหาฝั่งโมเดล -> 500
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Inference error.") from exc

    if not np.isfinite(fraud_probability):
        logger.error("Model returned a non-finite fraud score: %s", fraud_probability)
        raise HTTPException(status_code=500, detail="Model returned a non-finite fraud score.")

    is_fraud = int(fraud_probability >= threshold)

    fraud_predictions_total.labels(label="fraud" if is_fraud else "not_fraud").inc()
    fraud_score.observe(fraud_probability)

    return {
        "is_fraud": is_fraud,
        "fraud_score": fraud_probability,
        "threshold": threshold,
        "model_version": model_version,
    }


@app.get("/metrics")
def metrics():
    if os.getenv("PROMETHEUS_MULTIPROC_DIR"):
        # หลาย worker (uvicorn --workers) แต่ละ process มีตัวนับของตัวเอง ต้องรวมจากไฟล์ใน multiproc dir
        registry = CollectorRegistry()
        prometheus_multiprocess.MultiProcessCollector(registry)
    else:
        registry = METRICS_REGISTRY
    return Response(content=generate_latest(registry), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health_check():
    ready = _is_ready()
    body = {
        "status": "healthy" if ready else "unhealthy",
        "model_loaded": model is not None,
        "model_uri": MODEL_URI,
        "model_version": model_version,
        "run_id": model_run_id,
        "threshold": threshold,
    }
    return JSONResponse(status_code=200 if ready else 503, content=body)
