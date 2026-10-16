"""Unit tests for the prediction API (run with: pytest)."""

from fastapi.testclient import TestClient

from app.main import MODEL_VERSION, app

client = TestClient(app)

VALID_PATIENT = {
    "age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03, "s1": -0.02,
    "s2": 0.03, "s3": -0.02, "s4": 0.02, "s5": 0.02, "s6": -0.001,
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_version": MODEL_VERSION}


def test_predict_returns_a_number():
    response = client.post("/predict", json=VALID_PATIENT)
    assert response.status_code == 200
    assert isinstance(response.json()["prediction"], float)


def test_missing_feature_returns_422():
    patient = {k: v for k, v in VALID_PATIENT.items() if k != "s6"}
    response = client.post("/predict", json=patient)
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "Invalid input"
    assert body["details"][0]["field"] == "s6"


def test_wrong_type_returns_422():
    patient = {**VALID_PATIENT, "age": "abc"}
    response = client.post("/predict", json=patient)
    assert response.status_code == 422
    assert response.json()["details"][0]["field"] == "age"
