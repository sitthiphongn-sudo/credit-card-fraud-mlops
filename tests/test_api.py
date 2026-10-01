import pytest
from fastapi.testclient import TestClient

import serving.app as api


class FakeModel:
    def __init__(self, p):
        self.p = p

    def predict(self, X):
        import numpy as np
        # คืนค่าความน่าจะเป็นเทียบกับ Threshold 0.5 สำหรับตัดสินใจ 0 หรือ 1
        return np.array([1 if self.p >= 0.5 else 0] * len(X))


@pytest.fixture()
def client(monkeypatch):
    # จำลองให้โมเดลพร้อมใช้งานตอนเทสต์ API
    monkeypatch.setattr(api, "model", FakeModel(0.1))
    with TestClient(api.app) as c:
        yield c


def payload(amount=5000.0):
    # ส่ง 30 features ตามที่ Schema ของ app.py กำหนด (Time, V1..V28, Amount)
    features = [0.0] * 29 + [amount]
    return {"features": features}


def test_health_check(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"
    assert r.json()["model_loaded"] is True


def test_health_degraded_without_model(client, monkeypatch):
    monkeypatch.setattr(api, "model", None)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is False


def test_predict_success(client):
    r = client.post("/predict", json=payload())
    assert r.status_code == 200
    data = r.json()
    assert "is_fraud" in data
    assert data["status"] == "success"


def test_bad_schema_returns_400(client):
    # ส่งข้อมูลผิดโครงสร้าง (ขาด features หรือส่งผิดประเภท)
    assert client.post("/predict", json={"Amount": 5}).status_code == 422


def test_invalid_feature_length_returns_400(client):
    # ส่ง features ไม่ครบ 30 ตัว (เช่น ส่งไปแค่ 5 ตัว)
    r = client.post("/predict", json={"features": [0.0, 1.0, 2.0, 3.0, 4.0]})
    assert r.status_code == 400


def test_no_model_returns_500(client, monkeypatch):
    monkeypatch.setattr(api, "model", None)
    r = client.post("/predict", json=payload())
    assert r.status_code == 500