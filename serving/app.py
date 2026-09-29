import os
import time

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel

# เปลี่ยนมาใช้การ import ตามโครงสร้างเดิมของโปรเจกต์
from src.fraud.validate import FEATURES, validate

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
            
        # 1. Validate โครงสร้างข้อมูลตามฟังก์ชันเดิมของทีม
        df = validate(pd.DataFrame(req.instances), with_target=False)[FEATURES]
        
        decisions = []
        probabilities = []
        
        # 2. เริ่มทำงานแบบ Cascade
        for i, row in df.iterrows():
            # กฎ Hard Rules (ถ้า Amount > 3000 ให้ปฏิเสธทันที)
            if "Amount" in row and row["Amount"] > 3000:
                decisions.append("decline")
                probabilities.append(1.0)
                DECISIONS.labels("decline").inc()
                continue
                
            # 3. ถ้าผ่าน Hard rules ค่อยให้โมเดลทำนาย
            row_df = pd.DataFrame([row])
            p = float(model.predict_proba(row_df)[:, 1][0])
            probabilities.append(round(p, 4))
            
            if p >= 0.97:
                decision = "decline"
            elif p >= 0.80:
                decision = "review"
            else:
                decision = "approve"
                
            decisions.append(decision)
            DECISIONS.labels(decision).inc()

        REQUESTS.labels("200").inc()
        return {"probabilities": probabilities, "decisions": decisions}
        
    except HTTPException:
        REQUESTS.labels("503").inc()
        raise
    except Exception as exc:
        REQUESTS.labels("400").inc()
        raise HTTPException(400, str(exc)) from exc
    finally:
        LATENCY.observe(time.perf_counter() - start)