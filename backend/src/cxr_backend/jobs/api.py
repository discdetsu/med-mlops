"""Public: upload → job, job status + prediction, display image and heatmaps.
Internal: the worker reports each job's result."""

import hashlib
import logging
import uuid
from pathlib import PurePath

from botocore.exceptions import BotoCoreError, ClientError
from django.conf import settings
from django.db import transaction
from django.http import HttpRequest, HttpResponse
from django.urls import reverse
from django.utils import timezone
from kombu.exceptions import OperationalError
from ninja import File, Router, Status
from ninja.files import UploadedFile

from cxr_backend import storage
from cxr_backend.jobs.models import InferenceJob, Prediction
from cxr_backend.jobs.queue import enqueue
from cxr_backend.jobs.schemas import Error, JobOut, PredictionOut, ResultAck, ResultIn
from cxr_backend.jobs.upload import CONTENT_TYPES, EXTENSIONS, sniff_format
from cxr_backend.monitoring.metrics import INFERENCE_LATENCY, JOB_RESULTS, JOBS_CREATED

logger = logging.getLogger(__name__)

router = Router(tags=["jobs"])
internal_router = Router(tags=["internal"])


@router.post("", response={202: JobOut, 400: Error, 413: Error, 415: Error, 503: Error})
def create_job(request: HttpRequest, file: File[UploadedFile]) -> Status[JobOut | Error]:
    """Store the upload in `inference/raw/`, record a job, and queue it for a worker."""
    size = file.size or 0
    if size == 0:
        return Status(400, Error(detail="empty file"))
    if size > settings.MAX_UPLOAD_BYTES:
        return Status(413, Error(detail=f"file is larger than {settings.MAX_UPLOAD_BYTES} bytes"))
    fmt = sniff_format(file.read(132))
    if fmt is None:
        return Status(415, Error(detail="expected a PNG, JPEG, or DICOM (Part 10) file"))

    file.seek(0)
    digest = hashlib.sha256()
    for chunk in file.chunks():
        digest.update(chunk)
    file.seek(0)

    job_id = uuid.uuid4()
    raw_key = f"raw/{job_id}{EXTENSIONS[fmt]}"
    try:
        storage.put_raw(raw_key, file, CONTENT_TYPES[fmt])
    except BotoCoreError, ClientError:
        logger.exception("storing upload %s failed", raw_key)
        return Status(503, Error(detail="object storage is unavailable; try again"))

    job = InferenceJob.objects.create(
        id=job_id,
        filename=PurePath(file.name or "upload").name[:255],
        format=fmt,
        size_bytes=size,
        sha256=digest.hexdigest(),
        raw_key=raw_key,
    )
    try:
        enqueue(str(job.id), raw_key)
    except OperationalError:
        logger.exception("queueing job %s failed", job.id)
        job.status, job.error, job.finished_at = (
            InferenceJob.Status.FAILED,
            "could not queue the job",
            timezone.now(),
        )
        job.save(update_fields=["status", "error", "finished_at"])
        return Status(503, Error(detail="the job queue is unavailable; try again"))

    JOBS_CREATED.labels(fmt).inc()
    return Status(202, job_out(job))


@router.get("/{uuid:job_id}", response={200: JobOut, 404: Error})
def get_job(request: HttpRequest, job_id: uuid.UUID) -> Status[JobOut | Error]:
    job = InferenceJob.objects.select_related("prediction").filter(id=job_id).first()
    if job is None:
        return Status(404, Error(detail="job not found"))
    return Status(200, job_out(job))


@router.get("/{uuid:job_id}/image", url_name="job_image", response={404: Error})
def get_image(request: HttpRequest, job_id: uuid.UUID) -> HttpResponse | Status[Error]:
    """The windowed, downscaled image the heatmaps are aligned to, as an 8-bit PNG."""
    prediction = Prediction.objects.filter(job_id=job_id).first()
    if prediction is None:
        return Status(404, Error(detail="no prediction for this job"))
    return _png(prediction.image_key)


@router.get("/{uuid:job_id}/heatmaps/{class_name}", url_name="job_heatmap", response={404: Error})
def get_heatmap(
    request: HttpRequest, job_id: uuid.UUID, class_name: str
) -> HttpResponse | Status[Error]:
    """One class's CAM as an 8-bit grayscale PNG (0 = no evidence, 255 = the map's peak)."""
    prediction = Prediction.objects.filter(job_id=job_id).first()
    if prediction is None or class_name not in prediction.heatmap_keys:
        return Status(404, Error(detail="no such heatmap"))
    return _png(prediction.heatmap_keys[class_name])


@internal_router.post("/results", response={200: ResultAck, 404: Error})
def post_result(request: HttpRequest, payload: ResultIn) -> Status[ResultAck | Error]:
    """Record a worker's result once. A redelivered job reports again; later reports are no-ops."""
    with transaction.atomic():
        job = InferenceJob.objects.select_for_update().filter(id=payload.job_id).first()
        if job is None:
            return Status(404, Error(detail="job not found"))
        if job.status != InferenceJob.Status.QUEUED:
            return Status(200, ResultAck(job_id=job.id, status=job.status, duplicate=True))

        if payload.prediction is not None:
            Prediction.objects.create(job=job, **payload.prediction.model_dump())
            job.status = InferenceJob.Status.SUCCEEDED
        else:
            job.status = InferenceJob.Status.FAILED
            job.error = payload.error[:2000] or "inference failed"
        job.finished_at = timezone.now()
        job.save(update_fields=["status", "error", "finished_at"])

    version = payload.prediction.model_version if payload.prediction else ""
    JOB_RESULTS.labels(job.status, version).inc()
    if payload.prediction is not None:
        INFERENCE_LATENCY.labels(version).observe(payload.prediction.latency_ms / 1000)
    return Status(200, ResultAck(job_id=job.id, status=job.status, duplicate=False))


def job_out(job: InferenceJob) -> JobOut:
    prediction = getattr(job, "prediction", None) if job.status == "succeeded" else None
    return JobOut(
        id=job.id,
        status=job.status,
        filename=job.filename,
        format=job.format,
        created_at=job.created_at,
        finished_at=job.finished_at,
        error=job.error,
        prediction=None if prediction is None else _prediction_out(prediction),
    )


def _prediction_out(p: Prediction) -> PredictionOut:
    job_id = str(p.job_id)
    return PredictionOut(
        model_name=p.model_name,
        model_version=p.model_version,
        preprocess_version=p.preprocess_version,
        mlflow_run_id=p.mlflow_run_id,
        dataset_rev=p.dataset_rev,
        classes=p.classes,
        probabilities=p.probabilities,
        thresholds=p.thresholds,
        positives=p.positives,
        latency_ms=p.latency_ms,
        image_url=reverse("api:job_image", kwargs={"job_id": job_id}),
        heatmap_urls={
            c: reverse("api:job_heatmap", kwargs={"job_id": job_id, "class_name": c})
            for c in p.classes
        },
    )


def _png(key: str) -> HttpResponse | Status[Error]:
    try:
        body = storage.get_inference_object(key)
    except ClientError:
        logger.warning("missing inference object %s", key)
        return Status(404, Error(detail="image not found"))
    response = HttpResponse(body, content_type="image/png")
    # Keys are per job and never rewritten.
    response["Cache-Control"] = "private, max-age=86400, immutable"
    return response
