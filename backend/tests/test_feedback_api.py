from collections.abc import Callable

import pytest
from django.test import Client

from cxr_backend.feedback.models import Feedback

from .conftest import FakeS3, upload

pytestmark = pytest.mark.django_db


def test_agree_records_the_predicted_positives(
    client: Client, finished_job: Callable[..., str]
) -> None:
    job_id = finished_job()
    response = client.post(
        f"/api/jobs/{job_id}/feedback", {"verdict": "agree"}, content_type="application/json"
    )

    assert response.status_code == 201
    assert response.json()["labels"] == {"Atelectasis": 1, "Cardiomegaly": 0, "Effusion": 1}
    feedback = Feedback.objects.get()
    assert feedback.model_version == "7"
    assert feedback.verdict == "agree"


def test_correct_records_the_reviewer_labels(
    client: Client, finished_job: Callable[..., str]
) -> None:
    job_id = finished_job()
    labels = {"Atelectasis": False, "Cardiomegaly": True, "Effusion": False}
    response = client.post(
        f"/api/jobs/{job_id}/feedback",
        {"verdict": "correct", "labels": labels, "comment": "cardiomegaly missed"},
        content_type="application/json",
    )

    assert response.status_code == 201
    feedback = Feedback.objects.get()
    assert feedback.labels == {"Atelectasis": 0, "Cardiomegaly": 1, "Effusion": 0}
    assert feedback.comment == "cardiomegaly missed"


@pytest.mark.parametrize(
    "labels",
    [None, {"Atelectasis": True}, {"Atelectasis": True, "Cardiomegaly": False, "Nodule": True}],
)
def test_correct_needs_every_class(
    client: Client, finished_job: Callable[..., str], labels: dict[str, bool] | None
) -> None:
    job_id = finished_job()
    response = client.post(
        f"/api/jobs/{job_id}/feedback",
        {"verdict": "correct", "labels": labels},
        content_type="application/json",
    )
    assert response.status_code == 422
    assert not Feedback.objects.exists()


def test_feedback_needs_a_finished_prediction(
    client: Client, s3: FakeS3, queued: list[tuple[str, str]]
) -> None:
    job_id = upload(client).json()["id"]
    response = client.post(
        f"/api/jobs/{job_id}/feedback", {"verdict": "agree"}, content_type="application/json"
    )
    assert response.status_code == 409


def test_feedback_for_unknown_job_is_404(client: Client) -> None:
    response = client.post(
        "/api/jobs/00000000-0000-0000-0000-000000000000/feedback",
        {"verdict": "agree"},
        content_type="application/json",
    )
    assert response.status_code == 404
