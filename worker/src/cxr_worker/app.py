"""Celery app for the inference worker.

    celery -A cxr_worker.app worker -Q inference

The pool is always `solo`: one job at a time in the main thread. A model swap from
`model.events` then runs between jobs by construction, the ONNX session is never shared
across threads, and scaling out means more worker containers, each loading the champion
once (`docker compose up --scale worker=N`).

Failure handling: transient errors (S3, backend) retry with backoff; an image that can't
be decoded, a timeout, or running out of retries reports the job as failed and raises, and
with `task_acks_on_failure_or_timeout=False` Celery then rejects the message without
requeueing, so RabbitMQ dead-letters it to `dlq` (policy `inference-dead-letter`).
"""

import logging
import os

from celery import Celery

from cxr_core.messaging import INFERENCE_QUEUE
from cxr_worker.events import ModelEvents

app = Celery("cxr_worker", include=["cxr_worker.tasks"])
app.conf.update(
    broker_url=os.environ.get("CELERY_BROKER_URL", "memory://"),
    broker_connection_retry_on_startup=True,
    worker_pool="solo",
    task_default_queue=INFERENCE_QUEUE,
    task_serializer="json",
    accept_content=["json"],
    task_ignore_result=True,
    # Ack only after the job is done, one job in flight per worker, and reject (→ DLQ)
    # rather than ack when a job fails for good.
    task_acks_late=True,
    task_acks_on_failure_or_timeout=False,
    worker_prefetch_multiplier=1,
    # A job cut off by a lost connection is redelivered and then failed as poison (tasks.py);
    # if it had already reported, the backend keeps that first result.
    worker_cancel_long_running_tasks_on_connection_loss=False,
    worker_send_task_events=False,
    # RabbitMQ 4 refuses transient non-exclusive queues, which remote control (and so the
    # `inspect ping` healthcheck) declares by default. Gossip and mingle are off for the
    # same reason (Dockerfile CMD); this stack doesn't use them.
    control_queue_exclusive=True,
)
app.steps["consumer"].add(ModelEvents)

# httpx logs every request at INFO; the worker's own log lines say what happened.
logging.getLogger("httpx").setLevel(logging.WARNING)
