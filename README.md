# full-stack-on-prem-cxr-mlops

Chest X-ray abnormality classification with per-class probabilities and a CAM heatmap, from versioned data to label-driven retraining, all on-prem in Docker Compose.

> **Not a medical device.** This is a portfolio project. It is not clinically validated and must not be used for diagnosis.

Status: **Phase 1, data + training flow.** The infrastructure runs and is tested, and `make seed && make train` produces a registered `@challenger` on CPU. Serving and the UI come in Phase 2. See [`implementation-plan_v2.md`](implementation-plan_v2.md) and [`architecture_v2.html`](architecture_v2.html).

## Quickstart (infrastructure + training, for now)

Requires Docker with Compose v2, [uv](https://docs.astral.sh/uv/), and GNU make.

```sh
make install          # uv workspace (Python 3.14) + pre-commit hooks
make up               # core profile: Postgres, RustFS, RabbitMQ, MLflow; waits until healthy
make test             # unit tests
make test-integration # RustFS, MLflow artifacts, DVC push/pull, RabbitMQ fanout, dataset tables
make seed             # NIH subset from Kaggle → validate → DVC add + push → Postgres
make train            # train → evaluate → ONNX → E2E parity → reference → register @challenger
make bootstrap-champion  # first model → @champion, seed bundle in dist/
make up PROFILE=full  # + Prefect, Label Studio, Prometheus, Alertmanager, Pushgateway, Grafana
make help             # all targets
```

`make seed` fetches 4,000 images (≈1.6 GB, about 30 minutes) one file at a time from the public [Kaggle mirror](https://www.kaggle.com/datasets/nih-chest-xrays/data) of NIH ChestX-ray14; no Kaggle token is needed. If you already have the full dataset unpacked, `make seed SEED_ARGS="--from-archive /path/to/dataset"` copies from it in seconds; `--from-archive` alone downloads the 45 GB archive once (see [`data/README.md`](data/README.md)). `make train` uses the `demo` profile (ResNet18 at 128 px), which trains in minutes on a laptop CPU; `make train CONFIG=full` uses DenseNet121 at 320 px. Both flows run without a Prefect server under `core`; with `PROFILE=full` they report to the Prefect UI. After a seed, commit the DVC pointers: `git add data/*.dvc && git commit`.

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
packages/cxr_core/     shared preprocess / CAM / metrics / bundle (numpy only); model factory behind [train]
backend/               Django + Ninja API (Phase 2)
worker/                Celery ONNX inference worker, no Django, no torch (Phase 2)
flows/                 Prefect flows: ingest, train (Phase 1); promotion, monitor later
  src/cxr_flows/configs/ training profiles: demo, full
frontend/              Svelte + Vite UI (Phase 2)
data/                  DVC-tracked datasets; only .dvc pointers are committed
scripts/               download_nih.py, bootstrap_champion.py, wait_healthy.sh
tests/integration/     tests against the running core stack
docs/adr/              architecture decision records
docs/model-card.md     data, classes, metrics, limitations
```

## Decisions

See [`docs/adr/`](docs/adr/README.md).

## License

Code: [MIT](LICENSE).

Data: NIH ChestX-ray14, provided by the NIH Clinical Center. The images are not redistributed here; `make seed` downloads them from the public Kaggle mirror, and you are bound by the dataset's own terms (see the [NIH release](https://nihcc.app.box.com/v/ChestXray-NIHCC)). If you use it, cite:

> Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases. IEEE CVPR 2017.
