"""Phase 4 on the backend: image metadata on jobs, embeddings on predictions, and drift reports
in the admin."""

import io
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import pydicom
import pytest
from django.contrib.auth.models import User
from django.db import connection
from django.test import Client
from pydicom.dataset import FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from cxr_backend.jobs.metadata import ImageMetadata, normalize
from cxr_backend.jobs.models import InferenceJob, Prediction
from cxr_backend.monitoring.admin import REPORT_CSP
from cxr_backend.monitoring.models import DriftReport

from .conftest import FakeS3, upload

pytestmark = pytest.mark.django_db


def _dicom(**tags: str) -> bytes:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = pydicom.Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    for key, value in tags.items():
        setattr(ds, key, value)
    buf = io.BytesIO()
    ds.save_as(buf, enforce_file_format=True)
    return buf.getvalue()


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (("m", "45", "pa"), ImageMetadata("M", 45, "PA")),
        (("Female", 7, " AP "), ImageMetadata("F", 7, "AP")),
        (("O", "045Y", "LL"), ImageMetadata("", 45, "LL")),
        ((None, "006M", None), ImageMetadata("", 0, "")),
        (("", "413", ""), ImageMetadata("", None, "")),  # NIH has a few ages > 400
        (("", "old", ""), ImageMetadata("", None, "")),
    ],
)
def test_normalize_uses_the_nih_vocabulary(args: tuple[Any, ...], expected: ImageMetadata) -> None:
    assert normalize(*args) == expected


def _upload_with(client: Client, body: bytes, name: str, **fields: str) -> Any:
    f = io.BytesIO(body)
    f.name = name
    return client.post("/api/jobs", {"file": f, **fields})


def test_upload_stores_metadata_from_form_fields(
    client: Client, s3: FakeS3, queued: list[tuple[str, str]]
) -> None:
    from .conftest import PNG

    body = _upload_with(
        client, PNG, "x.png", patient_sex="F", patient_age="61", view_position="AP"
    ).json()
    assert body["metadata"] == {"patient_sex": "F", "patient_age": 61, "view_position": "AP"}
    job = InferenceJob.objects.get(id=body["id"])
    assert (job.patient_sex, job.patient_age, job.view_position) == ("F", 61, "AP")


def test_png_without_fields_has_no_metadata(
    client: Client, s3: FakeS3, queued: list[tuple[str, str]]
) -> None:
    assert upload(client).json()["metadata"] == {
        "patient_sex": "",
        "patient_age": None,
        "view_position": "",
    }


def test_dicom_header_fills_what_the_form_leaves_out(
    client: Client, s3: FakeS3, queued: list[tuple[str, str]]
) -> None:
    dicom = _dicom(PatientSex="M", PatientAge="052Y", ViewPosition="PA")
    body = _upload_with(client, dicom, "scan.dcm", view_position="AP").json()

    assert body["format"] == "dicom"
    assert body["metadata"] == {"patient_sex": "M", "patient_age": 52, "view_position": "AP"}
    # The whole file is still stored, not just what follows the header.
    assert s3.objects[("inference", queued[0][1])] == dicom


def test_unreadable_dicom_header_still_uploads(
    client: Client, s3: FakeS3, queued: list[tuple[str, str]]
) -> None:
    from .conftest import DICOM

    response = upload(client, DICOM, name="broken.dcm")
    assert response.status_code == 202
    assert response.json()["metadata"]["view_position"] == ""


def _dicom_header_end(body: bytes) -> int:
    """Where the file meta group ends: its length is the UL value at byte 140."""
    return 144 + int.from_bytes(body[140:144], "little")


def _malformed_headers() -> dict[str, bytes]:
    body = _dicom(ViewPosition="AP")
    meta = body[: _dicom_header_end(body)]
    modality = b"\x08\x00\x60\x00CS\x02\x00CR"
    nested = b""
    for _ in range(3000):
        nested = (
            b"\x08\x00\x40\x11SQ\x00\x00\xff\xff\xff\xff"
            + b"\xfe\xff\x00\xe0\xff\xff\xff\xff"
            + nested
            + b"\xfe\xff\x0d\xe0\x00\x00\x00\x00"
            + b"\xfe\xff\xdd\xe0\x00\x00\x00\x00"
        )
    return {
        "group length cut short": body[:141],
        "file meta cut short": body[:152],
        "unknown VR on a tag we read": meta + modality + b"\x18\x00\x01\x51C\x85\x02\x00AP",
        "sequences nested too deep": meta + nested,
    }


@pytest.mark.parametrize("case", list(_malformed_headers()))
def test_malformed_dicom_header_still_uploads(
    client: Client, s3: FakeS3, queued: list[tuple[str, str]], case: str
) -> None:
    """Whatever pydicom raises, the job is created and the worker decides what the file is."""
    response = upload(client, _malformed_headers()[case], name="broken.dcm")
    assert response.status_code == 202
    assert response.json()["metadata"] == {
        "patient_sex": "",
        "patient_age": None,
        "view_position": "",
    }
    assert len(queued) == 1


def test_jobs_insert_without_the_metadata_columns() -> None:
    """Code from before Phase 4 doesn't set them, so a rollback needs database defaults."""
    job_id = uuid.uuid4()
    with connection.cursor() as cur:
        cur.execute(
            """INSERT INTO inference_jobs (id, status, filename, format, size_bytes, sha256,
                                           raw_key, error, created_at)
               VALUES (%s, 'queued', 'x.png', 'png', 1, 'x', 'k', '', %s)""",
            [job_id.hex, datetime.now(UTC)],
        )
    job = InferenceJob.objects.get(id=job_id)
    assert (job.patient_sex, job.patient_age, job.view_position) == ("", None, "")


def test_result_stores_the_embedding(finished_job: Callable[..., str]) -> None:
    job_id = finished_job(embedding=[0.5, 1.25, 0.0])
    assert Prediction.objects.get(job_id=job_id).embedding == [0.5, 1.25, 0.0]


def test_result_without_embedding_is_still_accepted(finished_job: Callable[..., str]) -> None:
    job_id = finished_job()
    assert Prediction.objects.get(job_id=job_id).embedding is None


# ---------------------------------------------------------------------------- admin


def _report(uri: str = "s3://artifacts/monitoring/cxr-classifier/2/20261101T000000Z.html") -> Any:
    now = datetime(2026, 11, 1, tzinfo=UTC)
    return DriftReport.objects.create(
        model_name="cxr-classifier",
        model_version="2",
        window_start=now,
        window_end=now,
        n_predictions=150,
        n_reference=3000,
        window_fingerprint="abc",
        drift_detected=True,
        embedding_drift=0.71,
        prediction_drift=0.4,
        metadata_drift=0.54,
        report={
            "scores": [
                {
                    "kind": "embedding",
                    "feature": "embedding",
                    "method": "model",
                    "n": 150,
                    "score": 0.71,
                    "threshold": 0.6,
                    "drifted": True,
                }
            ],
            "delayed_auroc": {
                "reported": True,
                "n": 40,
                "min_n": 30,
                "macro": 0.8,
                "macro_ci": [0.7, 0.9],
                "n_agree": 10,
            },
        },
        report_uri=uri,
    )


@pytest.fixture
def staff(client: Client) -> Client:
    user = User.objects.create_superuser("admin", "a@example.com", "pw")
    client.force_login(user)
    return client


def test_admin_lists_and_shows_drift_reports(staff: Client) -> None:
    report = _report()
    listing = staff.get("/admin/monitoring/driftreport/")
    assert listing.status_code == 200
    assert f"/admin/monitoring/driftreport/{report.pk}/report/" in listing.content.decode()

    page = staff.get(f"/admin/monitoring/driftreport/{report.pk}/change/").content.decode()
    assert "0.710" in page
    assert "macro 0.800 (95% CI 0.700 to 0.900), n = 40, 10 of them &quot;agree&quot;" in page


def test_admin_streams_the_report_sandboxed(staff: Client, s3: FakeS3) -> None:
    report = _report()
    s3.objects[("artifacts", "monitoring/cxr-classifier/2/20261101T000000Z.html")] = b"<html>"
    response = staff.get(f"/admin/monitoring/driftreport/{report.pk}/report/")

    assert response.status_code == 200
    assert response.content == b"<html>"
    assert response["Content-Security-Policy"] == REPORT_CSP
    assert "sandbox allow-scripts" in REPORT_CSP and "allow-same-origin" not in REPORT_CSP


def test_admin_refuses_a_report_outside_the_versions_folder(staff: Client, s3: FakeS3) -> None:
    report = _report(uri="s3://artifacts/models/cxr-classifier/2/model.onnx")
    s3.objects[("artifacts", "models/cxr-classifier/2/model.onnx")] = b"weights"
    assert staff.get(f"/admin/monitoring/driftreport/{report.pk}/report/").status_code == 404


def test_report_needs_a_login(client: Client) -> None:
    report = _report()
    response = client.get(f"/admin/monitoring/driftreport/{report.pk}/report/")
    assert response.status_code == 302
