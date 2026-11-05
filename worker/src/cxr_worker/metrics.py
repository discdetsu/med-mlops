"""Prometheus metrics for the worker, served on WORKER_METRICS_PORT (default 9100).

The solo pool runs every job in the main process, so the default in-process registry is
complete. Prometheus finds every worker container through Docker's DNS (`--scale worker=N`),
and an absent `up{job="worker"} == 1` is the missing-worker alert.
"""

import logging
import os

from prometheus_client import Counter, Gauge, Histogram, start_http_server

logger = logging.getLogger(__name__)

JOBS = Counter(
    "cxr_worker_jobs_total",
    "Jobs by outcome: succeeded, failed (reported as failed, then dead-lettered), retried",
    ["outcome"],
)
JOB_DURATION = Histogram(
    "cxr_worker_job_duration_seconds",
    "Download → preprocess → ONNX → CAM → upload, per job that succeeded",
    ["model_version"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)
MODEL_INFO = Gauge(
    "cxr_worker_model_info",
    "1 for the model version this worker serves; absent before a champion is loaded",
    ["model_name", "model_version"],
)
MODEL_SWAPS = Counter(
    "cxr_worker_model_swaps_total", "Model loads after startup by result: ok, failed", ["result"]
)


def serving(model_name: str, model_version: str) -> None:
    MODEL_INFO.clear()
    MODEL_INFO.labels(model_name, model_version).set(1)


def start_server() -> None:
    """Called once from the worker's `worker_init` signal; WORKER_METRICS_PORT=0 turns it off."""
    port = int(os.environ.get("WORKER_METRICS_PORT", "9100"))
    if port:
        start_http_server(port)
        logger.info("metrics on :%d/metrics", port)
