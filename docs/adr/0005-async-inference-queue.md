# 0005: Asynchronous inference through RabbitMQ, on solo-pool Celery workers that swap models between jobs

- Status: Accepted
- Date: 2026-10-29

## Context

An upload has to turn into probabilities and heatmaps without holding an HTTP request open for decoding, ONNX inference, and CAM rendering, which take 50 to 300 ms on a laptop CPU and seconds for a large DICOM. The worker must stay free of Django and torch (ADR 0004). When `@champion` moves, every worker has to switch to the new version without dropping or splitting a job, and a job that can never succeed has to end up somewhere it can be inspected, not in a retry loop.

## Decision

- **Upload → queue → callback.** `POST /api/jobs` stores the file in `inference/raw/`, records an `inference_jobs` row, and publishes the task by name (`cxr_worker.infer`) to the `inference` queue; the backend never imports worker code. The worker posts the result to `POST /internal/results`, and the backend writes the `predictions` row. There is no Celery result backend: Postgres is the only record of a job.
- **Solo pool, one job at a time per worker container.** Scaling out means more containers (`docker compose up --scale worker=N`), each loading the champion once.
- **Model swaps happen between jobs.** Each worker binds an exclusive, server-named queue to the `model.events` fanout exchange through a Celery consumer bootstep. Its callback runs in the same event loop as the solo pool's jobs, so a swap can only happen between jobs, and replacing one reference is the whole swap. On startup and after every reconnect, the step asks `GET /internal/champion` first, so events missed while disconnected are not lost.
- **Internal API auth: a shared bearer token, plus the network.** `/internal/*` requires `INTERNAL_API_TOKEN` (compared in constant time), and Nginx doesn't route `/internal` or `/metrics` at all, so they are reachable only on the compose network. This settles the plan's open decision: both, not one or the other.
- **Failures.** Messages are acked late with a prefetch of 1. Transient errors (S3, the backend, no champion yet) retry with exponential backoff, up to 5 times. A poison job is reported as failed and then raised: an image that can't be decoded, a missing upload, the job timeout, running out of retries, or a message redelivered after a worker died mid-job. With `task_acks_on_failure_or_timeout=False`, Celery rejects the message without requeueing, and RabbitMQ's `inference-dead-letter` policy routes it to `dlq`.
- **Job timeout is SIGALRM in the worker**, because Celery's time limits need the prefork pool.
- **RabbitMQ 4 compatibility.** RabbitMQ 4 refuses transient non-exclusive queues, so remote control uses exclusive queues (`control_queue_exclusive`), which also keeps the `inspect ping` healthcheck working, and gossip and mingle are off.

## Consequences

- Each worker container handles one job at a time. A demo needs one or two; throughput scales linearly with containers, and so does RAM (about 150 MB plus the model per worker).
- A hung job makes the worker unhealthy, because its event loop also answers the healthcheck ping.
- A message redelivered after a dropped connection is failed rather than retried, even when the worker didn't actually crash. The user uploads again, and a result already reported for that job is kept, because the backend records only the first result.
- Messages Celery can't decode are logged and dropped by Celery, not dead-lettered. Only JSON is accepted, and the only publisher is the backend.
- The app tables use explicit names (`inference_jobs`, `predictions`, `feedback`) rather than Django's app-label prefix, so the monitor flow's SQL reads naturally. Tables the flows own must not reuse these names.

## Alternatives considered

- **Synchronous inference in Django:** puts onnxruntime in the backend, ties up gunicorn threads, and every backend process would need its own model swap logic.
- **Prefork pool:** Celery's time limits work, but every child process loads its own model, and swapping means restarting the pool or coordinating children, which drops jobs or holds two models in memory.
- **Celery result backend:** a second store for results that the backend would still have to copy into Postgres to keep the lineage and serve the UI.
- **Polling MLflow for alias changes instead of `model.events`:** slower to converge, and adds load on MLflow for every worker. The fanout event is also what Phase 3's approve and rollback actions will publish.
