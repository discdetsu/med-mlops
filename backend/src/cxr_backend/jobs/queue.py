"""Publishes inference jobs to RabbitMQ by task name; the backend never imports worker code."""

from functools import cache

from celery import Celery
from django.conf import settings

from cxr_core.messaging import INFER_TASK, INFERENCE_QUEUE


@cache
def _celery() -> Celery:
    app = Celery("cxr_backend", broker=settings.CELERY_BROKER_URL)
    app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        broker_connection_retry_on_startup=True,
        # Fail an upload fast instead of hanging the request when RabbitMQ is down.
        broker_connection_timeout=5,
        broker_transport_options={"max_retries": 2},
    )
    return app


def enqueue(job_id: str, raw_key: str) -> None:
    _celery().send_task(INFER_TASK, args=[job_id, raw_key], queue=INFERENCE_QUEUE)
