from typing import Literal

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2048)

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("URL cannot be empty or whitespace.")
        return value


class PredictResponse(BaseModel):
    url: str
    prediction: Literal["PHISHING", "LEGITIMATE"]
    phishing_probability: float = Field(ge=0, le=1)
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    model_version: str
    features: dict[str, float | int]


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
    model_version: str | None = None
