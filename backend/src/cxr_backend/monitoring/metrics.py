"""Prometheus metrics for the backend, scraped from /metrics (Phase 4).

Gunicorn runs one worker process with threads (see backend/Dockerfile), so the default
in-process registry is complete and no multiprocess directory is needed.
"""

from prometheus_client import Counter, Histogram

HTTP_REQUEST_DURATION = Histogram(
    "cxr_http_request_duration_seconds",
    "HTTP request latency by route pattern",
    ["method", "route", "status"],
)
JOBS_CREATED = Counter("cxr_jobs_created_total", "Uploads accepted and queued", ["format"])
JOB_RESULTS = Counter(
    "cxr_job_results_total", "Results reported by workers", ["status", "model_version"]
)
INFERENCE_LATENCY = Histogram(
    "cxr_inference_latency_seconds",
    "Worker time per job (download → preprocess → ONNX → CAM)",
    ["model_version"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)
FEEDBACK = Counter("cxr_feedback_total", "Feedback submissions", ["verdict", "model_version"])
