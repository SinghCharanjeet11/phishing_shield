from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from src.api.schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
)
from src.training.model_io import load_model


LOW_RISK_LIMIT = 0.4
HIGH_RISK_LIMIT = 0.7


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = None
    app.state.metadata = None
    app.state.model_error = None

    try:
        app.state.model, app.state.metadata = load_model()
        print("Model loaded successfully.")
    except Exception as exc:
        app.state.model_error = str(exc)
        print(f"Model unavailable: {exc}")

    yield


app = FastAPI(
    title="PhishingShield API",
    description="API for phishing URL detection.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def home():
    return {
        "message": "Welcome to PhishingShield API",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
def health(request: Request):
    metadata = request.app.state.metadata

    return HealthResponse(
        status="ok",
        model_loaded=request.app.state.model is not None,
        model_version=str(metadata["version"]) if metadata else None,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, request: Request):
    model = request.app.state.model
    metadata = request.app.state.metadata

    if model is None or metadata is None:
        raise HTTPException(
            status_code=503,
            detail="Trained model is not available. Please try again later.",
        )

    try:
        from src.features.extractor import extract_features
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="URL feature extractor is not available yet.",
        )

    try:
        features = extract_features(payload.url)
        feature_names = metadata["features"]

        missing = [name for name in feature_names if name not in features]
        if missing:
            raise ValueError(f"Missing model features: {missing}")

        feature_frame = pd.DataFrame(
            [[features[name] for name in feature_names]],
            columns=feature_names,
        )

        phishing_probability = float(model.predict_proba(feature_frame)[0][1])

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        )

    threshold = float(metadata.get("threshold", 0.5))

    prediction = (
        "PHISHING"
        if phishing_probability >= threshold
        else "LEGITIMATE"
    )

    if phishing_probability < LOW_RISK_LIMIT:
        risk_level = "LOW"
    elif phishing_probability <= HIGH_RISK_LIMIT:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"

    return PredictResponse(
        url=payload.url,
        prediction=prediction,
        phishing_probability=phishing_probability,
        risk_level=risk_level,
        model_version=str(metadata["version"]),
        features=features,
    )


@app.get("/model-info")
def model_info(request: Request):
    metadata = request.app.state.metadata

    if metadata is None:
        raise HTTPException(
            status_code=503,
            detail="Model metadata is not available.",
        )

    return metadata
