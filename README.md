# Lung CXR MLOps

Multi-label chest X-ray classification (NIH ChestX-ray14) with the full MLOps loop:
ETL, training, ONNX serving with CAM heatmaps, drift monitoring, and CI/CD.
See [`implementation-plan.md`](implementation-plan.md) and `Diagram v2.png`.

> Not a diagnostic tool.

## Layout

| Path | What |
| --- | --- |
| `src/cxr/` | Shared Python package: config, DB models, preprocessing |
| `migrations/` | Alembic migrations for the `app` database |
| `services/gateway`, `services/worker` | FastAPI gateway and inference workers (Phase 4) |
| `frontend/` | Svelte UI (Phase 4) |
| `dags/` | Airflow DAGs, mounted into the Airflow containers |
| `infra/` | Dockerfiles, Postgres init, helper scripts |
| `tests/` | pytest suite |

## Quickstart

Requires Docker with Compose and [uv](https://docs.astral.sh/uv/).

```bash
infra/scripts/init-env.sh     # writes .env with random secrets (or copy .env.example)
docker compose up -d
uv sync
uv run alembic upgrade head   # creates the dataset, serving, monitoring schemas
uv run pre-commit install
```

| UI | URL | Login |
| --- | --- | --- |
| MLflow | http://localhost:5000 | none |
| Airflow | http://localhost:8080 | none (local dev: everyone is admin) |
| RabbitMQ | http://localhost:15672 | `RABBITMQ_USER` / `RABBITMQ_PASSWORD` |
| RustFS console | http://localhost:9001/rustfs/console/ | `CXR_S3_ACCESS_KEY` / `CXR_S3_SECRET_KEY` |

Postgres (`localhost:5432`) holds three databases, each owned by its own role:
`app`, `mlflow`, `airflow`. RustFS has two buckets: `cxr` (data) and `mlflow` (artifacts).

## Tests

```bash
uv run pytest                                  # unit tests; DB tests are skipped

# Migration tests need a throwaway database (create it once):
docker compose exec postgres createdb -U postgres -O app app_test
CXR_TEST_DATABASE_URL=postgresql+psycopg://app:<CXR_DB_PASSWORD>@localhost:5432/app_test \
    uv run pytest
```

The migration tests drop and recreate the schemas in the target database, so point
`CXR_TEST_DATABASE_URL` at a throwaway database, never `app`.
