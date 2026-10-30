from collections.abc import Callable

import pytest
from django.test import Client, override_settings
from kombu.exceptions import OperationalError

import cxr_backend.jobs.api as jobs_api
from cxr_backend.jobs.models import InferenceJob

from .conftest import CLASSES, DICOM, PNG, FakeS3, post_json, prediction_payload, upload

pytestmark = pytest.mark.django_db


def test_upload_stores_raw_file_and_queues_the_job(
    client: Client, s3: FakeS3, queued: list[tuple[str, str]]
) -> None:
    response = upload(client)

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert body["format"] == "png"
    assert body["prediction"] is None
    raw_key = f"raw/{body['id']}.png"
    assert s3.objects[("inference", raw_key)] == PNG
    assert queued == [(body["id"], raw_key)]
    job = InferenceJob.objects.get(id=body["id"])
    assert job.filename == "cxr.png"
    assert job.size_bytes == len(PNG)


def test_dicom_is_recognized_by_content_not_name(
    client: Client, s3: FakeS3, queued: list[tuple[str, str]]
) -> None:
    body = upload(client, DICOM, name="scan.png").json()
    assert body["format"] == "dicom"
    assert queued[0][1].endswith(".dcm")


@pytest.mark.parametrize(
    ("content", "status"),
    [(b"", 400), (b"GIF89a" + b"\0" * 200, 415)],
)
def test_rejects_empty_and_unknown_files(
    client: Client, s3: FakeS3, queued: list[tuple[str, str]], content: bytes, status: int
) -> None:
    assert upload(client, content).status_code == status
    assert not queued
    assert not InferenceJob.objects.exists()


@override_settings(MAX_UPLOAD_BYTES=100)
def test_rejects_files_over_the_limit(
    client: Client, s3: FakeS3, queued: list[tuple[str, str]]
) -> None:
    assert upload(client).status_code == 413


def test_queue_outage_fails_the_job_and_returns_503(
    client: Client, s3: FakeS3, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(job_id: str, key: str) -> None:
        raise OperationalError("connection refused")

    monkeypatch.setattr(jobs_api, "enqueue", broken)
    assert upload(client).status_code == 503
    job = InferenceJob.objects.get()
    assert job.status == "failed"
    assert job.error == "could not queue the job"


def test_unknown_job_is_404(client: Client) -> None:
    assert client.get("/api/jobs/00000000-0000-0000-0000-000000000000").status_code == 404


def test_finished_job_returns_prediction_with_lineage(
    client: Client, finished_job: Callable[..., str]
) -> None:
    job_id = finished_job()
    body = client.get(f"/api/jobs/{job_id}").json()

    assert body["status"] == "succeeded"
    p = body["prediction"]
    assert p["model_version"] == "7"
    assert p["mlflow_run_id"] == "run123"
    assert p["dataset_rev"] == "f774f7a855ab"
    assert p["classes"] == CLASSES
    # 0.3 >= 0.3: a probability at the threshold counts as positive
    assert p["positives"] == ["Atelectasis", "Effusion"]
    assert p["image_url"] == f"/api/jobs/{job_id}/image"
    assert p["heatmap_urls"]["Effusion"] == f"/api/jobs/{job_id}/heatmaps/Effusion"


def test_image_and_heatmaps_are_streamed_from_storage(
    client: Client, s3: FakeS3, finished_job: Callable[..., str]
) -> None:
    job_id = finished_job()
    s3.objects[("inference", "results/x/image.png")] = b"image-bytes"
    s3.objects[("inference", "results/x/cam/1.png")] = b"cam-bytes"

    image = client.get(f"/api/jobs/{job_id}/image")
    assert image.status_code == 200
    assert image["Content-Type"] == "image/png"
    assert image.content == b"image-bytes"
    assert client.get(f"/api/jobs/{job_id}/heatmaps/Cardiomegaly").content == b"cam-bytes"
    assert client.get(f"/api/jobs/{job_id}/heatmaps/Nodule").status_code == 404
    assert client.get(f"/api/jobs/{job_id}/heatmaps/Atelectasis").status_code == 404  # not stored


def test_internal_api_requires_the_token(client: Client, finished_job: Callable[..., str]) -> None:
    payload = {"job_id": finished_job(), "status": "failed", "error": "x"}
    assert post_json(client, "/internal/results", payload).status_code == 401
    wrong = Client(headers={"Authorization": "Bearer nope"})
    assert post_json(wrong, "/internal/results", payload).status_code == 401
    assert wrong.get("/internal/champion").status_code == 401


def test_a_result_is_recorded_once(
    client: Client, internal: Client, finished_job: Callable[..., str]
) -> None:
    job_id = finished_job()
    prediction = prediction_payload(model_version="8")
    again = {"job_id": job_id, "status": "succeeded", "prediction": prediction}
    response = post_json(internal, "/internal/results", again)

    assert response.status_code == 200
    assert response.json()["duplicate"] is True
    assert client.get(f"/api/jobs/{job_id}").json()["prediction"]["model_version"] == "7"


def test_failed_result_marks_the_job_failed(
    client: Client, internal: Client, s3: FakeS3, queued: list[tuple[str, str]]
) -> None:
    job_id = upload(client).json()["id"]
    payload = {"job_id": job_id, "status": "failed", "error": "BadImage: cannot identify image"}
    assert post_json(internal, "/internal/results", payload).status_code == 200

    body = client.get(f"/api/jobs/{job_id}").json()
    assert body["status"] == "failed"
    assert body["error"] == "BadImage: cannot identify image"
    assert body["finished_at"] is not None


@pytest.mark.parametrize(
    "payload",
    [
        {"status": "succeeded"},  # no prediction
        {"status": "failed", "prediction": prediction_payload()},
        {"status": "succeeded", "prediction": prediction_payload(thresholds={"Atelectasis": 0.5})},
        {"status": "succeeded", "prediction": prediction_payload(classes=[])},
    ],
)
def test_malformed_results_are_rejected(
    client: Client,
    internal: Client,
    s3: FakeS3,
    queued: list[tuple[str, str]],
    payload: dict[str, object],
) -> None:
    job_id = upload(client).json()["id"]
    response = internal.post(
        "/internal/results", {"job_id": job_id, **payload}, content_type="application/json"
    )
    assert response.status_code == 422
    assert InferenceJob.objects.get(id=job_id).status == "queued"


def test_result_for_unknown_job_is_404(internal: Client) -> None:
    payload = {"job_id": "00000000-0000-0000-0000-000000000000", "status": "failed"}
    assert post_json(internal, "/internal/results", payload).status_code == 404


def test_health(client: Client) -> None:
    assert client.get("/api/health").json() == {"status": "ok"}


def test_metrics_count_jobs_and_results(client: Client, finished_job: Callable[..., str]) -> None:
    finished_job()
    text = client.get("/metrics").content.decode()
    assert 'cxr_jobs_created_total{format="png"}' in text
    assert 'cxr_job_results_total{model_version="7",status="succeeded"}' in text
    assert "cxr_inference_latency_seconds_bucket" in text
    assert 'route="api/jobs"' in text
