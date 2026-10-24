# 0002: Two Compose profiles, `core` and `full`

- Status: Accepted
- Date: 2026-10-24

## Context

The full stack (serving, Prefect, Label Studio, Prometheus, Alertmanager, Grafana, Pushgateway) needs more RAM than a typical reviewer's laptop has spare. Most visitors only want to see an upload turn into a heatmap. Plan v1 split the extras into `monitoring` and `labeling` profiles, but the labeling loop needs monitoring (drift selects what to label), so those two profiles were never useful on their own.

## Decision

Two profiles only:

| Profile | Services | Purpose |
| --- | --- | --- |
| `core` | Postgres, RustFS, RabbitMQ, MLflow; Nginx, web UI, Django, and the Celery worker from Phase 2 | Upload → heatmap demo and promotion in Django admin, with a seeded `@champion` and no training |
| `full` | `core` + Prefect, Label Studio, Pushgateway, Prometheus, Alertmanager, Grafana | The label → retrain → gate → promote → monitor loop |

Every service lists its profiles explicitly (`[core, full]` or `[full]`), has a healthcheck, and uses `depends_on: condition: service_healthy` for its dependencies. One-shot `*-init` jobs (buckets, RabbitMQ topology) are idempotent and use `service_completed_successfully`.

`make up` waits with `scripts/wait_healthy.sh`, not `docker compose up --wait`, because `--wait` reports any exited container as a failure, including init jobs that exited 0.

## Consequences

- `core` is the main path in the README and in CI. `full` gets measured RAM numbers in the README so people know what they are getting into.
- No service can run outside a profile, so a plain `docker compose up` starts nothing. That is intentional: the Makefile always names a profile.
- Features that only make sense in `full` (drift alerts, Label Studio tasks) must degrade quietly in `core`; for example, the backend must not fail when Pushgateway is absent.

## Alternatives considered

- **v1 split (`core` / `monitoring` / `labeling`):** more combinations to test, and the useful ones were always "core" or "everything".
- **Separate compose files with `-f` overlays:** works, but duplicates service definitions and is easier to get wrong than one file with profiles.
