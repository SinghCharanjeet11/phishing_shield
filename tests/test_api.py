from fastapi.testclient import TestClient

from src.api.main import app


def test_home():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["message"] == "Welcome to PhishingShield API"


def test_health_without_model():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.json()["model_loaded"] is False
        assert response.json()["model_version"] is None


def test_predict_without_model():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"url": "https://example.com"},
        )
        assert response.status_code == 503


def test_model_info_without_model():
    with TestClient(app) as client:
        response = client.get("/model-info")
        assert response.status_code == 503


def test_predict_rejects_empty_url():
    with TestClient(app) as client:
        response = client.post("/predict", json={"url": ""})
        assert response.status_code == 422


def test_predict_rejects_whitespace_url():
    with TestClient(app) as client:
        response = client.post("/predict", json={"url": "   "})
        assert response.status_code == 422


def test_predict_rejects_missing_url():
    with TestClient(app) as client:
        response = client.post("/predict", json={})
        assert response.status_code == 422


def test_predict_rejects_long_url():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"url": "a" * 2049},
        )
        assert response.status_code == 422
