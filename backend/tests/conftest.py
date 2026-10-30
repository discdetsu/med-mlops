"""API contract tests run against SQLite (cxr_backend.settings_test) with S3, the job queue,
and MLflow replaced by in-memory fakes."""

import io
from collections.abc import Callable
from typing import IO, Any

import pytest
from django.conf import settings
from django.test import Client

import cxr_backend.jobs.api as jobs_api
from cxr_backend import storage

CLASSES = ["Atelectasis", "Cardiomegaly", "Effusion"]
PNG = b"\x89PNG\r\n\x1a\n" + b"\0" * 200
DICOM = b"\0" * 128 + b"DICM" + b"\0" * 200


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}

    def put_raw(self, key: str, body: IO[bytes], content_type: str) -> None:
        self.objects[(settings.INFERENCE_BUCKET, key)] = body.read()

    def get_inference_object(self, key: str) -> bytes:
        from botocore.exceptions import ClientError

        try:
            return self.objects[(settings.INFERENCE_BUCKET, key)]
        except KeyError:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject") from None

    def upload_file(self, path: str, bucket: str, key: str) -> None:
        with open(path, "rb") as f:
            self.objects[(bucket, key)] = f.read()


@pytest.fixture
def s3(monkeypatch: pytest.MonkeyPatch) -> FakeS3:
    fake = FakeS3()
    for name in ("put_raw", "get_inference_object", "upload_file"):
        monkeypatch.setattr(storage, name, getattr(fake, name))
    return fake


@pytest.fixture
def queued(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(jobs_api, "enqueue", lambda job_id, key: calls.append((job_id, key)))
    return calls


@pytest.fixture
def internal() -> Client:
    return Client(headers={"Authorization": f"Bearer {settings.INTERNAL_API_TOKEN}"})


def post_json(client: Client, url: str, payload: object) -> Any:
    return client.post(url, payload, content_type="application/json")


def upload(client: Client, body: bytes = PNG, name: str = "cxr.png") -> Any:
    f = io.BytesIO(body)
    f.name = name
    return client.post("/api/jobs", {"file": f})


def prediction_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model_name": "cxr-classifier",
        "model_version": "7",
        "preprocess_version": "1",
        "mlflow_run_id": "run123",
        "dataset_rev": "f774f7a855ab",
        "classes": CLASSES,
        "probabilities": {"Atelectasis": 0.8, "Cardiomegaly": 0.1, "Effusion": 0.3},
        "thresholds": {"Atelectasis": 0.5, "Cardiomegaly": 0.5, "Effusion": 0.3},
        "image_key": "results/x/image.png",
        "heatmap_keys": {c: f"results/x/cam/{k}.png" for k, c in enumerate(CLASSES)},
        "latency_ms": 42.0,
        "worker": "worker-1",
    }
    payload.update(overrides)
    return payload


@pytest.fixture
def finished_job(
    db: None, client: Client, internal: Client, s3: FakeS3, queued: list[tuple[str, str]]
) -> Callable[..., str]:
    """Upload a file and report a successful result for it; returns the job id."""

    def make(**overrides: Any) -> str:
        job_id: str = upload(client).json()["id"]
        prediction = prediction_payload(**overrides)
        result = {"job_id": job_id, "status": "succeeded", "prediction": prediction}
        assert post_json(internal, "/internal/results", result).status_code == 200
        return job_id

    return make
