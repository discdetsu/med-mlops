# full-stack-on-prem-cxr-mlops

Chest X-ray abnormality classification with per-class probabilities and a CAM heatmap, from versioned data to label-driven retraining, all on-prem in Docker Compose.

> **Not a medical device.** This is a portfolio project. It is not clinically validated and must not be used for diagnosis.

Status: **Phase 3, registry + promotion.** Upload a chest X-ray in the browser and get per-class probabilities and a class activation map from the `@champion` model, without training anything. Training (`make seed && make train`) produces new versions on CPU, a gate compares each one with `@champion` on the locked test set, and a person approves it in Django admin before every worker switches to it. See [`implementation-plan_v2.md`](implementation-plan_v2.md) and [`architecture_v2.html`](architecture_v2.html).

## Quickstart: upload → heatmap

Requires Docker with Compose v2 and GNU make.

```sh
make up           # core profile: builds and starts everything, waits until healthy
make seed-model   # registers the released seed model as @champion; workers load it
open http://localhost:8000
```

Upload a frontal chest X-ray (PNG, JPEG, or DICOM). The page shows the image with a heatmap for the selected finding, the probability of each of the five findings against its threshold, a feedback form, and which model version, dataset rev, and MLflow run produced the result. Django admin is at http://localhost:8000/admin/ (`DJANGO_ADMIN_USER` / `DJANGO_ADMIN_PASSWORD` from `.env`). `make smoke` runs the same flow with curl.

`make seed-model` downloads `SEED_BUNDLE` (a GitHub Release asset, checked against its `.sha256`) and registers it in the local MLflow. The `seed-v2` release isn't published yet, so for now use a bundle you built with `make bootstrap-champion`: `make seed-model SEED_BUNDLE=dist/cxr-seed-bundle-v2.tar.gz`. When a `@champion` already exists it does nothing; add `SEED_MODEL_ARGS=--force` to register the bundle anyway.

How a request flows: Nginx → Django stores the upload in RustFS and queues a job on RabbitMQ → a Celery worker (ONNX Runtime, no torch) runs `cxr_core.preprocess` → ONNX → CAM and posts the result back → the UI polls for it. When `@champion` moves, a `model.events` fanout tells every worker to load the new version between jobs. Jobs that can never succeed go to the `dlq` queue. See [ADR 0005](docs/adr/0005-async-inference-queue.md).

## Training and development

Also requires [uv](https://docs.astral.sh/uv/).

```sh
make install          # uv workspace (Python 3.14) + pre-commit hooks
make test             # unit + API contract tests
make test-integration # against `make up`: RustFS, MLflow, DVC, RabbitMQ, app tables, gateway
make seed             # NIH subset from Kaggle → validate → DVC add + push → Postgres
make train            # train → evaluate → ONNX → E2E parity → reference → @challenger → gate
make gate VERSION=N   # re-run the promotion gate for a version (default: @challenger)
make bootstrap-champion  # first model → @champion, seed bundle in dist/
make up PROFILE=full  # + Prefect, Label Studio, Prometheus, Alertmanager, Pushgateway, Grafana
make help             # all targets
```

`make seed` fetches 4,000 images (≈1.6 GB, about 30 minutes) one file at a time from the public [Kaggle mirror](https://www.kaggle.com/datasets/nih-chest-xrays/data) of NIH ChestX-ray14; no Kaggle token is needed. If you already have the full dataset unpacked, `make seed SEED_ARGS="--from-archive /path/to/dataset"` copies from it instead (seconds for the default subset; about 17 minutes for all 112k images with `--positives-per-class all --negatives all`, most of it `dvc push`); `--from-archive` alone downloads the 45 GB archive once (see [`data/README.md`](data/README.md)). `make train` uses the `demo` profile (ResNet18 at 128 px, trained on a fixed 4,000-image sample of the train split): about 30 minutes on a laptop CPU with the full dataset pinned, most of it scoring the 16k val images every epoch and the 16.6k `test_locked` images at the end. `make train CONFIG=full` uses DenseNet121 at 320 px; its float32 tensor cache needs about 1.2 MB per image, so on the full dataset it doesn't fit on most disks (about 140 GB). Both flows run without a Prefect server under `core`; with `PROFILE=full` they report to the Prefect UI. After a seed, commit the DVC pointers: `git add data/*.dvc && git commit`.

The committed pointers pin the full NIH dataset, and its files live only in the DVC remote they were pushed to: the `datasets` bucket on the RustFS of the machine that seeded them. A fresh clone starts with an empty RustFS, so `make seed` and `make train` stop with `PinnedDataMissingError`, because the frozen `test_locked` can't be regenerated. Either copy that bucket's contents in from the machine that has them, or delete `data/*.dvc` and run `make seed` to start a dataset of your own, with its own locked test set (its models are only comparable with each other).

## Promotion: gate → approve → every worker switches

`make train` ends with the promotion gate: the new `@challenger` and the current `@champion` score the same 16.6k-image locked test set through their ONNX bundles, and a paired bootstrap over patients puts a CI on each AUROC difference. The gate passes when macro AUROC is non-inferior (lower 95% bound of the difference ≥ −0.02), no sex, view, or age subgroup drops by more than 0.02, ECE rises by at most 0.01, and E2E parity passed; the thresholds live in [`gate.yaml`](flows/src/cxr_flows/configs/gate.yaml). It takes about 3 minutes on a laptop CPU. See [ADR 0007](docs/adr/0007-human-promotion-gate.md).

The result waits in Django admin under **Registry → Model candidates**, with the full report. **Approve** (needs a reason) moves `@champion` to the new version and the old one to `@previous`, then tells every worker to load it after its current job. **Reject** closes the candidate. **Roll back** on the list page moves `@champion` back to `@previous`. Each decision is logged in **Model approvals**. A candidate can only be approved while `@champion` is still the version it was compared with; otherwise `make gate VERSION=N` re-runs the comparison. The same actions work without the admin: `make promote VERSION=N REASON="..."` and `make rollback REASON="..."`. With `auto_promote: true` in `gate.yaml`, `make train` approves a passing version by itself (for the scripted demo).

Checking that a promotion reaches every worker:

```sh
docker compose --profile core up -d --scale worker=2
make gate VERSION=N   # or make train; then approve version N in the admin
make smoke            # repeat a few times: each prediction must come from version N
                      # then roll back in the admin
make smoke            # predictions come from the previous champion again
```

`make up` copies `.env.example` to `.env` on first run. Change the passwords there for anything other than a local demo.

| Service | URL | Profile |
| --- | --- | --- |
| Web UI and API (Nginx) | http://localhost:8000 | core |
| Django admin | http://localhost:8000/admin/ | core |
| MLflow | http://localhost:5000 | core |
| RustFS console | http://localhost:9001 | core |
| RabbitMQ management | http://localhost:15672 | core |
| Prefect | http://localhost:4200 | full |
| Label Studio | http://localhost:8080 | full |
| Grafana | http://localhost:3000 | full |
| Prometheus | http://localhost:9090 | full |
| Alertmanager | http://localhost:9093 | full |

All ports bind to `127.0.0.1` only. Django has no published port: browsers reach it through Nginx, and its `/internal` API (for workers) and `/metrics` stay on the compose network.

## Repository layout

```
compose/               config mounted into compose services (one folder per service)
  postgres/init/         creates app, mlflow, prefect, labelstudio DBs, one user each
  rustfs/init.sh         buckets + inference/raw/ lifecycle rule
  rabbitmq/              plugins, model.events fanout, dlx → dlq, dead-letter policy
  mlflow/                MLflow server image (Postgres backend, RustFS artifacts)
  nginx/                 gateway: UI, /api, /admin; rate and upload limits
  prometheus/ alertmanager/ grafana/
packages/cxr_core/     shared preprocess / ONNX inference / CAM / metrics / bundle / registry names
                       (numpy only); model factory behind [train]
backend/               Django + Ninja: jobs, feedback, registry (champion, seed_model, promotion
                       admin), monitoring
worker/                Celery ONNX inference worker, no Django, no torch
flows/                 Prefect flows: ingest, train, promotion (gate); monitor later
  src/cxr_flows/configs/ training profiles (demo, full) and gate.yaml
frontend/              Svelte + Vite UI, built into the nginx image
data/                  DVC-tracked datasets; only .dvc pointers are committed
scripts/               download_nih.py, bootstrap_champion.py, smoke_core.sh, wait_healthy.sh
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
