"""API contract tests run against SQLite (cxr_backend.settings_test) with S3, the job queue,
and MLflow replaced by in-memory fakes."""

import io
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import IO, Any

import pytest
from django.conf import settings
from django.test import Client
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, RESOURCE_DOES_NOT_EXIST

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

    def get_object(self, bucket: str, key: str) -> bytes:
        from botocore.exceptions import ClientError

        try:
            return self.objects[(bucket, key)]
        except KeyError:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject") from None


class FakeMlflow:
    """The slice of MlflowClient that seed(), current_champion(), and promotion use."""

    def __init__(self, champion: str | None = None) -> None:
        self.versions: dict[str, dict[str, str]] = {}
        self.aliases: dict[str, str] = {}
        self.artifacts: dict[str, bytes] = {}
        self.run_status: str | None = None
        # ("set" | "delete", alias) pairs whose next call raises, as a flaky server would.
        self.fail: set[tuple[str, str]] = set()
        if champion is not None:
            self.versions[champion] = {
                "bundle_uri": f"s3://artifacts/models/cxr-classifier/{champion}/"
            }
            self.aliases["champion"] = champion

    def get_registered_model(self, name: str) -> Any:
        if not self.versions:
            raise MlflowException("no model", error_code=RESOURCE_DOES_NOT_EXIST)
        return SimpleNamespace(name=name, aliases=dict(self.aliases))

    def get_model_version(self, name: str, version: str) -> Any:
        return SimpleNamespace(version=version, tags=self.versions[version], run_id="run-x")

    def get_model_version_by_alias(self, name: str, alias: str) -> Any:
        # Like the real server: an unknown alias on an existing model is not "does not exist".
        if not self.versions:
            raise MlflowException("no model", error_code=RESOURCE_DOES_NOT_EXIST)
        if alias not in self.aliases:
            raise MlflowException(f"alias {alias} not found", error_code=INVALID_PARAMETER_VALUE)
        return self.get_model_version(name, self.aliases[alias])

    def get_experiment_by_name(self, name: str) -> None:
        return None

    def create_experiment(self, name: str) -> str:
        return "1"

    def create_run(self, experiment_id: str, run_name: str, tags: dict[str, str]) -> Any:
        info = SimpleNamespace(
            run_id="seed-run", artifact_uri="mlflow-artifacts:/1/seed-run/artifacts"
        )
        return SimpleNamespace(info=info)

    def search_registered_models(self, filter_string: str) -> list[object]:
        return []

    def create_registered_model(self, name: str) -> None:
        pass

    def create_model_version(self, name: str, source: str, run_id: str, description: str) -> Any:
        version = str(len(self.versions) + 1)
        self.versions[version] = {}
        return SimpleNamespace(version=version)

    def log_artifacts(self, run_id: str, local_dir: str, artifact_path: str) -> None:
        for path in Path(local_dir).iterdir():
            self.artifacts[f"{artifact_path}/{path.name}"] = path.read_bytes()

    def set_model_version_tag(self, name: str, version: str, key: str, value: str) -> None:
        self.versions[version][key] = value

    def set_registered_model_alias(self, name: str, alias: str, version: str) -> None:
        self._maybe_fail("set", alias)
        self.aliases[alias] = version

    def delete_registered_model_alias(self, name: str, alias: str) -> None:
        self._maybe_fail("delete", alias)
        del self.aliases[alias]

    def _maybe_fail(self, op: str, alias: str) -> None:
        if (op, alias) in self.fail:
            self.fail.discard((op, alias))
            raise MlflowException(f"injected failure: {op} @{alias}")

    def set_terminated(self, run_id: str, status: str = "FINISHED") -> None:
        self.run_status = status


@pytest.fixture
def s3(monkeypatch: pytest.MonkeyPatch) -> FakeS3:
    fake = FakeS3()
    for name in ("put_raw", "get_inference_object", "upload_file", "get_object"):
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
