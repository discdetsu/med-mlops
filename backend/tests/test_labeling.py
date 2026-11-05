"""Phase 5 on the backend: the patient ID on uploads, the labeling loop's rows in the admin, and
auto-promote for a train run started from a Prefect deployment."""

import io
import uuid
from collections.abc import Callable
from typing import Any

import pytest
from django.contrib.auth.models import User
from django.db import connection
from django.test import Client
from mlflow.exceptions import MlflowException

import cxr_backend.registry.promotion as promotion
from cxr_backend.jobs import metadata
from cxr_backend.jobs.models import InferenceJob
from cxr_backend.labeling.models import LabelingTask, RetrainDecision
from cxr_backend.registry.models import ModelApproval, ModelCandidate

from .conftest import PNG, FakeMlflow, FakeS3, post_json
from .test_monitoring import _dicom

pytestmark = pytest.mark.django_db


def _upload(client: Client, body: bytes = PNG, **fields: str) -> Any:
    f = io.BytesIO(body)
    f.name = "scan"
    return client.post("/api/jobs", {"file": f, **fields})


def test_upload_stores_the_patient_id_but_never_returns_it(
    client: Client, s3: FakeS3, queued: list[tuple[str, str]]
) -> None:
    response = _upload(client, patient_id=" 00000042 ")
    assert response.status_code == 202
    assert InferenceJob.objects.get(id=response.json()["id"]).patient_id == "00000042"
    assert "00000042" not in response.content.decode()


def test_dicom_header_fills_in_the_patient_id_unless_the_form_gives_one(
    client: Client, s3: FakeS3, queued: list[tuple[str, str]]
) -> None:
    dicom = _dicom(PatientID="P-17")
    from_header = _upload(client, dicom).json()["id"]
    from_form = _upload(client, dicom, patient_id="form").json()["id"]
    assert InferenceJob.objects.get(id=from_header).patient_id == "P-17"
    assert InferenceJob.objects.get(id=from_form).patient_id == "form"


def test_a_patient_id_that_is_too_long_is_dropped_not_cut() -> None:
    assert metadata.normalize(patient_id="x" * 64).patient_id == "x" * 64
    assert metadata.normalize(patient_id="x" * 65).patient_id == ""


def test_jobs_insert_without_the_patient_id_column() -> None:
    """Code from before Phase 5 doesn't set it, so a rollback needs a database default."""
    job_id = uuid.uuid4()
    with connection.cursor() as cur:
        cur.execute(
            """INSERT INTO inference_jobs (id, status, filename, format, size_bytes, sha256,
                                           raw_key, error, created_at)
               VALUES (%s, 'queued', 'x.png', 'png', 1, 'x', 'k', '', CURRENT_TIMESTAMP)""",
            [job_id.hex],
        )
    assert InferenceJob.objects.get(id=job_id).patient_id == ""


def test_the_flows_insert_tasks_relying_on_the_database_defaults(
    finished_job: Callable[..., str],
) -> None:
    job_id = uuid.UUID(finished_job())
    with connection.cursor() as cur:
        cur.execute(
            """INSERT INTO labeling_tasks (sha256, job_id, model_name, model_version, reason,
                                           ls_task_id)
               VALUES ('abc', %s, 'cxr-classifier', '2', 'drift', 7)""",
            [job_id.hex],
        )
    task = LabelingTask.objects.get()
    assert (task.status, task.image_id, task.quarantine_reason) == ("open", "", "")
    assert task.created_at is not None


@pytest.fixture
def staff(client: Client) -> Client:
    client.force_login(User.objects.create_superuser("admin", "a@example.com", "pw"))
    return client


def test_admin_lists_tasks_and_decisions_read_only(
    staff: Client, finished_job: Callable[..., str]
) -> None:
    LabelingTask.objects.create(
        sha256="abc",
        job_id=finished_job(),
        model_name="cxr-classifier",
        model_version="2",
        reason="disagreement",
        ls_task_id=7,
        status="quarantined",
        quarantine_reason="patient is in test_locked",
    )
    RetrainDecision.objects.create(
        dataset_rev="newrev",
        baseline_rev="oldrev",
        new_labels=3,
        min_new_labels=50,
        triggered=False,
        reason="3 new labels, at least 50 needed",
    )
    tasks = staff.get("/admin/labeling/labelingtask/")
    assert tasks.status_code == 200
    assert "patient is in test_locked" in tasks.content.decode()
    decisions = staff.get("/admin/labeling/retraindecision/")
    assert decisions.status_code == 200
    assert "at least 50 needed" in decisions.content.decode()

    assert staff.get("/admin/labeling/labelingtask/add/").status_code == 403
    pk = LabelingTask.objects.get().pk
    assert staff.post(f"/admin/labeling/labelingtask/{pk}/change/", {}).status_code == 403


def test_auto_promote_needs_the_internal_token(client: Client) -> None:
    response = post_json(client, "/internal/auto-promote", {"version": "3"})
    assert response.status_code == 401


def _candidate(version: str, **fields: Any) -> ModelCandidate:
    defaults: dict[str, Any] = {
        "model_name": "cxr-classifier",
        "model_version": version,
        "champion_version": "1",
        "status": ModelCandidate.Status.PENDING,
        "gate_passed": True,
        "report": {},
        "report_uri": "",
        "mlflow_run_id": "",
        "dataset_rev": "r",
    }
    return ModelCandidate.objects.create(**(defaults | fields))


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch) -> FakeMlflow:
    fake = FakeMlflow(champion="1")
    fake.versions["3"] = {}
    monkeypatch.setattr(promotion, "mlflow_client", lambda: fake)
    monkeypatch.setattr(promotion, "publish_champion_changed", lambda event: None)
    return fake


def test_auto_promote_approves_a_marked_candidate(internal: Client, registry: FakeMlflow) -> None:
    _candidate("3", auto_promote=True)
    response = post_json(internal, "/internal/auto-promote", {"version": "3"})
    assert response.status_code == 200
    assert response.json() == {
        "promoted": True, "from_version": "1", "to_version": "3", "event_published": True,
    }  # fmt: skip
    assert registry.aliases["champion"] == "3"
    assert ModelApproval.objects.get().actor == promotion.AUTO_PROMOTE_ACTOR


def test_auto_promote_is_a_no_op_for_an_unmarked_candidate(
    internal: Client, registry: FakeMlflow
) -> None:
    _candidate("3", auto_promote=False)
    response = post_json(internal, "/internal/auto-promote", {"version": "3"})
    assert response.status_code == 200 and response.json()["promoted"] is False
    assert registry.aliases["champion"] == "1"
    assert not ModelApproval.objects.exists()


def test_auto_promote_refuses_a_stale_comparison(internal: Client, registry: FakeMlflow) -> None:
    _candidate("3", auto_promote=True, champion_version="9")
    response = post_json(internal, "/internal/auto-promote", {"version": "3"})
    assert response.status_code == 409
    assert "re-run the gate" in response.json()["detail"]
    assert registry.aliases["champion"] == "1"


def test_auto_promote_reports_a_registry_outage(
    internal: Client, registry: FakeMlflow, monkeypatch: pytest.MonkeyPatch
) -> None:
    def down(version: str) -> None:
        raise MlflowException("down")

    monkeypatch.setattr(promotion, "auto_promote", down)
    response = post_json(internal, "/internal/auto-promote", {"version": "3"})
    assert response.status_code == 503
