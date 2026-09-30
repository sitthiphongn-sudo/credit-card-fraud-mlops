import os
from contextlib import asynccontextmanager

import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

LOCAL_MODEL_DIR = "./models_cache/champion_model"
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if not os.path.exists(LOCAL_MODEL_DIR):
        mlflow.artifacts.download_artifacts(
            artifact_uri="models:/fraud-detector@champion",
            dst_path=LOCAL_MODEL_DIR
        )
    model = mlflow.pyfunc.load_model(LOCAL_MODEL_DIR)
    yield

# สร้าง App เพียวๆ โดยไม่มี Middleware มารั้งความเร็ว
app = FastAPI(title="Credit Card Fraud Detection API", lifespan=lifespan)

FEATURE_COLUMNS = [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
]

class TransactionData(BaseModel):
    features: list[float] = Field(..., description="List of 30 numerical features")

@app.post("/predict")
def predict(data: TransactionData):  # ใช้ def ธรรมดา เพื่อกระจายงานลง ThreadPool
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")
    
    if len(data.features) != 30:
        raise HTTPException(status_code=400, detail="Expected 30 features.")

    try:
        # สร้าง DataFrame ด้วยวิธีที่เร็วที่สุด
        input_df = pd.DataFrame([data.features], columns=FEATURE_COLUMNS)
        
        prediction = model.predict(input_df)
        
        raw_val = prediction[0] if isinstance(prediction, (np.ndarray, list, pd.Series)) else prediction
        is_fraud_res = int(raw_val.item()) if hasattr(raw_val, 'item') else int(raw_val)
            
        return {"is_fraud": is_fraud_res, "status": "success"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}") from e

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}