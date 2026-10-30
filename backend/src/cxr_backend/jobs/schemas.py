from datetime import datetime
from typing import Literal
from uuid import UUID

from ninja import Schema
from pydantic import model_validator


class PredictionOut(Schema):
    model_name: str
    model_version: str
    preprocess_version: str
    mlflow_run_id: str
    dataset_rev: str
    classes: list[str]
    probabilities: dict[str, float]
    thresholds: dict[str, float]
    positives: list[str]
    latency_ms: float
    image_url: str
    heatmap_urls: dict[str, str]


class JobOut(Schema):
    id: UUID
    status: str
    filename: str
    format: str
    created_at: datetime
    finished_at: datetime | None
    error: str
    prediction: PredictionOut | None


class Error(Schema):
    detail: str


class PredictionIn(Schema):
    model_name: str
    model_version: str
    preprocess_version: str
    mlflow_run_id: str = ""
    dataset_rev: str = ""
    classes: list[str]
    probabilities: dict[str, float]
    thresholds: dict[str, float]
    image_key: str
    heatmap_keys: dict[str, str]
    latency_ms: float
    worker: str = ""

    @model_validator(mode="after")
    def _keys_match_classes(self) -> PredictionIn:
        classes = set(self.classes)
        if len(classes) != len(self.classes) or not classes:
            raise ValueError("classes must be non-empty and unique")
        for name in ("probabilities", "thresholds", "heatmap_keys"):
            if set(getattr(self, name)) != classes:
                raise ValueError(f"{name} keys must equal classes")
        return self


class ResultIn(Schema):
    job_id: UUID
    status: Literal["succeeded", "failed"]
    error: str = ""
    prediction: PredictionIn | None = None

    @model_validator(mode="after")
    def _prediction_iff_succeeded(self) -> ResultIn:
        if (self.status == "succeeded") != (self.prediction is not None):
            raise ValueError(
                "a succeeded result needs a prediction, a failed one must not have one"
            )
        return self


class ResultAck(Schema):
    job_id: UUID
    status: str
    duplicate: bool
