"""Unit tests ของ Serving API (สหรัฐ) — ไม่ต้องมี MLflow server จริง

ใช้ FakeModel แทน champion และจำลอง MlflowClient เพื่อตรวจว่า
โมเดล/threshold/model_version มาจาก alias champion ของ Registry
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

import serving.app as api

TEST_THRESHOLD = 0.51  # ค่า threshold ของ champion ที่ทีมตกลง (lightgbm_none)
REQUIRED_METRICS = (
    "fraud_requests_total",
    "fraud_request_latency_seconds",
    "fraud_predictions_total",
    "fraud_score",
)


class FakeModel:
    """คืน probability ของ class fraud ตามที่กำหนด และจำ input ล่าสุดไว้ตรวจ"""

    def __init__(self, fraud_probability):
        self.fraud_probability = fraud_probability
        self.last_input = None

    def predict_proba(self, X):
        self.last_input = X
        p = float(self.fraud_probability)
        return np.array([[1.0 - p, p]] * len(X))

    def predict(self, X):  # ถ้า API เรียก predict() แทน predict_proba ให้เทสต์ล้ม
        raise AssertionError("API must use predict_proba, not predict")


@pytest.fixture()
def client(monkeypatch):
    def fake_load_champion_model():
        api.model = FakeModel(0.70)
        api.threshold = TEST_THRESHOLD
        api.model_version = "7"
        api.model_run_id = "run-7"

    monkeypatch.setattr(api, "load_champion_model", fake_load_champion_model)

    with TestClient(api.app) as test_client:
        yield test_client

    api.model = api.threshold = api.model_version = api.model_run_id = None


def payload(amount=5000.0, n=30):
    return {"features": [0.0] * (n - 1) + [amount]}


def raw_post(client, body: str):
    """ส่ง body ตรง ๆ (เช่น NaN ซึ่ง json.dumps แบบเข้มงวดของ httpx ส่งไม่ได้) แบบเดียวกับ curl"""
    return client.post("/predict", content=body, headers={"Content-Type": "application/json"})


def metric(name, **labels):
    value = api.METRICS_REGISTRY.get_sample_value(name, labels)
    return 0.0 if value is None else value


# ---------------------------------------------------------------------------
# โหลด champion จาก MLflow Registry
# ---------------------------------------------------------------------------
def test_load_champion_uses_registry_alias_and_run_threshold(monkeypatch):
    loaded_uris = []
    champion = SimpleNamespace(version="3", run_id="3db9851cb0b64f1e8987cc009d3d49ab")

    class FakeClient:
        def get_model_version_by_alias(self, name, alias):
            assert (name, alias) == ("fraud-detector", "champion")
            return champion

        def get_run(self, run_id):
            assert run_id == champion.run_id
            return SimpleNamespace(data=SimpleNamespace(params={"threshold": "0.51"}))

    def fake_load_model(uri):
        loaded_uris.append(uri)
        return FakeModel(0.2)

    monkeypatch.setattr(api, "MlflowClient", FakeClient)
    # patch ด้วย path เต็ม: mlflow.sklearn เป็น lazy module ถ้า patch ผ่าน api.mlflow.sklearn
    # จะโดนตัว LazyLoader แทน แล้วเทสต์จะไปโหลดโมเดลจริงจาก ./mlruns โดยไม่รู้ตัว
    monkeypatch.setattr("mlflow.sklearn.load_model", fake_load_model)
    monkeypatch.setattr(api, "model", None)
    monkeypatch.setattr(api, "threshold", None)
    monkeypatch.setattr(api, "model_version", None)
    monkeypatch.setattr(api, "model_run_id", None)

    api.load_champion_model()

    assert loaded_uris == ["models:/fraud-detector@champion"]
    assert api.threshold == pytest.approx(0.51)
    assert api.model_version == "3"
    assert api.model_run_id == champion.run_id


def test_load_champion_fails_without_threshold_param(monkeypatch):
    class FakeClient:
        def get_model_version_by_alias(self, name, alias):
            return SimpleNamespace(version="1", run_id="abc")

        def get_run(self, run_id):
            return SimpleNamespace(data=SimpleNamespace(params={}))

    monkeypatch.setattr(api, "MlflowClient", FakeClient)
    monkeypatch.setattr("mlflow.sklearn.load_model", lambda uri: FakeModel(0.2))
    monkeypatch.setattr(api, "model", None)

    with pytest.raises(RuntimeError, match="threshold"):
        api.load_champion_model()

    assert api.model is None


def test_model_uri_is_registry_champion_not_local_cache():
    assert api.MODEL_URI == "models:/fraud-detector@champion"


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------
def test_health_check(client):
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model_version"] == "7"
    assert data["threshold"] == TEST_THRESHOLD
    assert data["model_uri"] == "models:/fraud-detector@champion"


@pytest.mark.parametrize("missing", ["model", "threshold", "model_version"])
def test_health_unhealthy_when_not_loaded(client, monkeypatch, missing):
    monkeypatch.setattr(api, missing, None)

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["status"] == "unhealthy"


# ---------------------------------------------------------------------------
# /predict ปกติ + threshold
# ---------------------------------------------------------------------------
def test_predict_normal_returns_contract_fields(client):
    before = metric("fraud_requests_total", status="success")

    response = client.post("/predict", json=payload())

    assert response.status_code == 200
    assert response.json() == {
        "is_fraud": 1,
        "fraud_score": 0.7,
        "threshold": TEST_THRESHOLD,
        "model_version": "7",
    }
    assert metric("fraud_requests_total", status="success") == before + 1


def test_predict_sends_30_named_columns_to_model(client):
    client.post("/predict", json=payload(amount=12.5))

    sent = api.model.last_input
    assert list(sent.columns) == ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
    assert sent.shape == (1, 30)
    assert sent["Amount"].iloc[0] == 12.5


@pytest.mark.parametrize(
    ("score", "expected_is_fraud", "expected_label"),
    [
        (0.51, 1, "fraud"),  # เท่ากับ threshold -> fraud (>=)
        (0.70, 1, "fraud"),
        (0.5099, 0, "not_fraud"),
        (0.50, 0, "not_fraud"),  # model.predict จะตัดที่ 0.5 แต่ข้อตกลงคือ 0.51
    ],
)
def test_threshold_behavior(client, monkeypatch, score, expected_is_fraud, expected_label):
    monkeypatch.setattr(api, "model", FakeModel(score))
    before = metric("fraud_predictions_total", label=expected_label)

    response = client.post("/predict", json=payload())

    assert response.status_code == 200
    body = response.json()
    assert body["threshold"] == TEST_THRESHOLD
    assert body["fraud_score"] == pytest.approx(score)
    assert body["is_fraud"] == expected_is_fraud
    assert metric("fraud_predictions_total", label=expected_label) == before + 1


def test_zero_amount_is_valid(client):
    assert client.post("/predict", json=payload(amount=0.0)).status_code == 200


# ---------------------------------------------------------------------------
# ข้อมูลผิดปกติ -> 400 และนับ status="bad_request"
# ---------------------------------------------------------------------------
def nan_body():
    return '{"features": [' + ", ".join(["0.0"] * 29) + ", NaN]}"


def test_nan_returns_400_and_counts_bad_request(client):
    before = metric("fraud_requests_total", status="bad_request")

    response = raw_post(client, nan_body())

    assert response.status_code == 400
    assert "finite" in response.json()["detail"]
    assert metric("fraud_requests_total", status="bad_request") == before + 1
    assert 'fraud_requests_total{status="bad_request"}' in client.get("/metrics").text


@pytest.mark.parametrize("token", ["Infinity", "-Infinity"])
def test_infinity_returns_400(client, token):
    body = '{"features": [' + ", ".join(["0.0"] * 29) + f", {token}]" + "}"
    assert raw_post(client, body).status_code == 400


def test_negative_amount_returns_400(client):
    before = metric("fraud_requests_total", status="bad_request")

    response = client.post("/predict", json=payload(amount=-1.0))

    assert response.status_code == 400
    assert "Amount" in response.json()["detail"]
    assert metric("fraud_requests_total", status="bad_request") == before + 1


@pytest.mark.parametrize(
    "body",
    [
        {"features": ["abc"]},
        {"features": [0.0] * 29 + ["not-a-number"]},
        {"features": [0.0] * 29 + ["1.5"]},
        {"features": ["abc"] + [0.0] * 29},
        {"features": [0.0] * 29 + [None]},
        {"features": [0.0] * 29 + [True]},
        {"features": [0.0] * 29 + [[1.0]]},
        {"features": "abc"},
    ],
)
def test_text_or_wrong_type_returns_400(client, body):
    before = metric("fraud_requests_total", status="bad_request")

    response = client.post("/predict", json=body)

    assert response.status_code == 400
    assert metric("fraud_requests_total", status="bad_request") == before + 1


@pytest.mark.parametrize("n", [0, 5, 29, 31])
def test_wrong_feature_count_returns_400(client, n):
    response = client.post("/predict", json={"features": [0.0] * n})

    assert response.status_code == 400
    assert "Expected 30 features" in response.json()["detail"]


@pytest.mark.parametrize(
    "body",
    [
        "{not json",
        "",
        json.dumps({"Amount": 5}),
        json.dumps([0.0] * 30),
    ],
)
def test_malformed_or_missing_body_returns_400(client, body):
    response = raw_post(client, body)

    assert response.status_code == 400


def test_huge_integer_returns_400_not_500(client):
    body = '{"features": [' + ", ".join(["0"] * 29) + ", " + "9" * 400 + "]}"
    assert raw_post(client, body).status_code == 400


# ---------------------------------------------------------------------------
# ฝั่ง server ผิดพลาด -> 5xx และนับ status="error"
# ---------------------------------------------------------------------------
def test_no_model_returns_503_and_counts_error(client, monkeypatch):
    monkeypatch.setattr(api, "model", None)
    before = metric("fraud_requests_total", status="error")

    response = client.post("/predict", json=payload())

    assert response.status_code == 503
    assert metric("fraud_requests_total", status="error") == before + 1


def test_model_failure_returns_500_and_counts_error(client, monkeypatch):
    class BrokenModel:
        def predict_proba(self, X):
            raise RuntimeError("boom")

    monkeypatch.setattr(api, "model", BrokenModel())
    before = metric("fraud_requests_total", status="error")

    response = client.post("/predict", json=payload())

    assert response.status_code == 500
    assert metric("fraud_requests_total", status="error") == before + 1


# ---------------------------------------------------------------------------
# /metrics
# ---------------------------------------------------------------------------
def test_metrics_endpoint_has_required_metric_families(client):
    client.post("/predict", json=payload())

    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    text = response.text
    assert "# TYPE fraud_requests_total counter" in text
    assert "# TYPE fraud_request_latency_seconds histogram" in text
    assert "# TYPE fraud_predictions_total counter" in text
    assert "# TYPE fraud_score histogram" in text
    for status in ("success", "bad_request", "error"):
        assert f'fraud_requests_total{{status="{status}"}}' in text
    for label in ("fraud", "not_fraud"):
        assert f'fraud_predictions_total{{label="{label}"}}' in text
    assert "fraud_request_latency_seconds_bucket" in text
    assert "fraud_score_bucket" in text
    # registry แยก: ไม่มี metric ของ process/python ปน
    assert "python_info" not in text
    assert "process_cpu_seconds_total" not in text


def test_latency_histogram_counts_every_predict_request(client):
    before = metric("fraud_request_latency_seconds_count")

    client.post("/predict", json=payload())
    client.post("/predict", json=payload(amount=-5.0))

    assert metric("fraud_request_latency_seconds_count") == before + 2


def test_health_and_metrics_are_not_counted_as_predict_requests(client):
    before = sum(
        metric("fraud_requests_total", status=s) for s in ("success", "bad_request", "error")
    )

    client.get("/health")
    client.get("/metrics")

    after = sum(
        metric("fraud_requests_total", status=s) for s in ("success", "bad_request", "error")
    )
    assert after == before
