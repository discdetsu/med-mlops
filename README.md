# full-stack-on-prem-cxr-mlops

Chest X-ray abnormality classification with per-class probabilities and a CAM heatmap, from versioned data to label-driven retraining, all on-prem in Docker Compose.

> **Not a medical device.** This is a portfolio project. It is not clinically validated and must not be used for diagnosis.

Status: **Phase 0, repo foundation.** The infrastructure runs and is tested; ML code, serving, and the UI come in later phases. See [`implementation-plan_v2.md`](implementation-plan_v2.md) and [`architecture_v2.html`](architecture_v2.html).

## Quickstart (infrastructure only, for now)

Requires Docker with Compose v2, [uv](https://docs.astral.sh/uv/), and GNU make.

```sh
make install          # uv workspace (Python 3.14) + pre-commit hooks
make up               # core profile: Postgres, RustFS, RabbitMQ, MLflow; waits until healthy
make test             # unit tests
make test-integration # RustFS, MLflow artifacts, DVC push/pull, RabbitMQ fanout
make up PROFILE=full  # + Prefect, Label Studio, Prometheus, Alertmanager, Pushgateway, Grafana
make help             # all targets
```

`make up` copies `.env.example` to `.env` on first run. Change the passwords there for anything other than a local demo.

| Service | URL | Profile |
| --- | --- | --- |
| MLflow | http://localhost:5000 | core |
| RustFS console | http://localhost:9001 | core |
| RabbitMQ management | http://localhost:15672 | core |
| Prefect | http://localhost:4200 | full |
| Label Studio | http://localhost:8080 | full |
| Grafana | http://localhost:3000 | full |
| Prometheus | http://localhost:9090 | full |
| Alertmanager | http://localhost:9093 | full |

All ports bind to `127.0.0.1` only.

## Repository layout

```
compose/               config mounted into compose services (one folder per service)
  postgres/init/         creates app, mlflow, prefect, labelstudio DBs, one user each
  rustfs/init.sh         buckets + inference/raw/ lifecycle rule
  rabbitmq/              plugins, model.events fanout, dlx → dlq, dead-letter policy
  mlflow/                MLflow server image (Postgres backend, RustFS artifacts)
  prometheus/ alertmanager/ grafana/
packages/cxr_core/     shared preprocess / CAM / metrics; numpy only, torch behind [train]
backend/               Django + Ninja API (Phase 2)
worker/                Celery ONNX inference worker, no Django, no torch (Phase 2)
flows/                 Prefect flows: ingest, train, promotion, monitor (Phase 1+)
frontend/              Svelte + Vite UI (Phase 2)
data/                  DVC-tracked datasets; only .dvc pointers are committed
scripts/               operational scripts (wait_healthy.sh; NIH download etc. later)
tests/integration/     tests against the running core stack
docs/adr/              architecture decision records
```

## Decisions

See [`docs/adr/`](docs/adr/README.md).

## License

Code: [MIT](LICENSE). Dataset terms and citation for NIH ChestX-ray14 will be added with the data pipeline (Phase 1).
