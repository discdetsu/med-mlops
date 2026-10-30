import logging
import socket
import tempfile
import time
from pathlib import Path, PurePath
from typing import Any, NoReturn

import httpx
from botocore.exceptions import BotoCoreError, ClientError
from celery import Task

from cxr_core.messaging import INFER_TASK
from cxr_worker.app import app
from cxr_worker.context import Context, context
from cxr_worker.deadline import JobTimeout, deadline
from cxr_worker.inference import BadImage, run_inference, to_png
from cxr_worker.model import LoadedModel, ModelUnavailable

logger = logging.getLogger(__name__)

MAX_RETRIES = 5
# Errors worth retrying: the backend, S3, or the registry being briefly unavailable.
TRANSIENT = (httpx.HTTPError, BotoCoreError, ClientError, ModelUnavailable, OSError)


class PoisonJob(Exception):
    """Raised to fail the task for good; Celery rejects the message and it goes to the DLQ."""


@app.task(name=INFER_TASK, bind=True, max_retries=MAX_RETRIES)  # type: ignore[untyped-decorator]
def infer(self: Task, job_id: str, raw_key: str) -> None:
    ctx = context()
    if (self.request.delivery_info or {}).get("redelivered"):
        # Delivered before and never acked: a worker died during this job. It may well be
        # the reason (out of memory on a huge image), so don't let it take down the next one.
        _fail(ctx, job_id, "a worker stopped while processing this job; not retrying it")
        raise PoisonJob(f"job {job_id} was redelivered")

    started = time.perf_counter()
    try:
        model = ctx.holder.require()
        with deadline(ctx.settings.job_timeout_s):
            payload = _process(ctx, model, job_id, raw_key)
    except (BadImage, JobTimeout) as exc:
        _fail(ctx, job_id, str(exc))
        raise PoisonJob(f"job {job_id}: {exc}") from exc
    except ClientError as exc:
        if _missing_object(exc):
            _fail(ctx, job_id, "the uploaded file is gone from storage")
            raise PoisonJob(f"job {job_id}: {raw_key} not found") from exc
        _retry_or_fail(self, ctx, job_id, exc)
    except TRANSIENT as exc:
        _retry_or_fail(self, ctx, job_id, exc)

    payload["latency_ms"] = (time.perf_counter() - started) * 1000
    try:
        ctx.backend.post_result({"job_id": job_id, "status": "succeeded", "prediction": payload})
    except httpx.HTTPError as exc:
        # Re-running the job is safe: object keys are per job and the backend records the
        # first result only.
        _retry_or_fail(self, ctx, job_id, exc)


def _process(ctx: Context, model: LoadedModel, job_id: str, raw_key: str) -> dict[str, Any]:
    bucket = ctx.settings.inference_bucket
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / PurePath(raw_key).name
        ctx.s3.download_file(bucket, raw_key, str(path))
        result = run_inference(model, path, ctx.settings.display_size)

    prefix = f"results/{job_id}/"
    image_key = prefix + "image.png"
    ctx.s3.put_object(
        Bucket=bucket, Key=image_key, Body=to_png(result.display), ContentType="image/png"
    )
    heatmap_keys = {}
    for k, (name, heatmap) in enumerate(result.heatmaps.items()):
        # Index, not class name, in the key: class names may hold characters keys shouldn't.
        key = f"{prefix}cam/{k}.png"
        ctx.s3.put_object(Bucket=bucket, Key=key, Body=to_png(heatmap), ContentType="image/png")
        heatmap_keys[name] = key

    manifest = model.manifest
    return {
        "model_name": manifest.model_name,
        "model_version": manifest.model_version,
        "preprocess_version": manifest.preprocess_version,
        "mlflow_run_id": manifest.lineage.get("mlflow_run_id", ""),
        "dataset_rev": manifest.lineage.get("dataset_rev", ""),
        "classes": list(manifest.classes),
        "probabilities": result.probabilities,
        "thresholds": dict(manifest.thresholds),
        "image_key": image_key,
        "heatmap_keys": heatmap_keys,
        "worker": socket.gethostname(),
    }


def _retry_or_fail(task: Task, ctx: Context, job_id: str, exc: Exception) -> NoReturn:
    retries = task.request.retries
    if retries >= MAX_RETRIES:
        _fail(ctx, job_id, f"gave up after {retries} retries: {type(exc).__name__}: {exc}")
        raise exc
    logger.warning("job %s: %s; retry %d of %d", job_id, exc, retries + 1, MAX_RETRIES)
    raise task.retry(exc=exc, countdown=min(2 ** (retries + 1), 60))


def _fail(ctx: Context, job_id: str, error: str) -> None:
    """Best effort: if the backend is down too, the job stays queued and the message is in
    the DLQ for inspection."""
    try:
        ctx.backend.post_result({"job_id": job_id, "status": "failed", "error": error})
    except httpx.HTTPError:
        logger.exception("could not report job %s as failed", job_id)


def _missing_object(exc: ClientError) -> bool:
    code = exc.response.get("Error", {}).get("Code")
    return code in ("404", "NoSuchKey")
