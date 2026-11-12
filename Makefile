PROFILE ?= core
COMPOSE := docker compose --profile $(PROFILE)

# Host-side S3 clients (tests, DVC, `make train`) reach RustFS on localhost.
-include .env
export AWS_ENDPOINT_URL := $(S3_HOST_ENDPOINT_URL)
export MLFLOW_S3_ENDPOINT_URL := $(S3_HOST_ENDPOINT_URL)
export AWS_ACCESS_KEY_ID := $(S3_ACCESS_KEY)
export AWS_SECRET_ACCESS_KEY := $(S3_SECRET_KEY)
export AWS_DEFAULT_REGION := $(S3_REGION)
export MLFLOW_TRACKING_URI := http://localhost:5000
# MLflow would hand out presigned URLs for http://rustfs:9000, which the host can't resolve.
export MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD := false
export MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD := false
export MLFLOW_DISABLE_AGENT_HINT := 1
export APP_DATABASE_URL := postgresql://app:$(APP_DB_PASSWORD)@localhost:5432/app
# Under core, Prefect runs flows against a temporary in-process server; under full they
# report to the Prefect server so runs show up in its UI.
ifeq ($(PROFILE),full)
export PREFECT_API_URL := http://localhost:4200/api
# `make monitor` pushes its scores like the scheduled `monitor` service does.
export PUSHGATEWAY_URL := http://localhost:9091
export LABEL_STUDIO_URL := http://localhost:8080
export LABEL_STUDIO_USERNAME
export LABEL_STUDIO_PASSWORD
endif
# flows-worker writes to the repo as this user (docker-compose.yml).
export HOST_UID := $(shell id -u)
export HOST_GID := $(shell id -g)

CONFIG ?= demo
SEED_ARGS ?=
TRAIN_ARGS ?=
GATE_ARGS ?=
MONITOR_ARGS ?=
LABEL_EXPORT_ARGS ?=
DEMO_ARGS ?=
ARGS ?=
PROMETHEUS_IMAGE := prom/prometheus:v3.15.1

# What `make seed-model` registers: a release URL (default) or a local path such as
# dist/cxr-seed-bundle-v2.tar.gz from `make bootstrap-champion`. Its .sha256 must sit next to it.
SEED_BUNDLE ?= https://github.com/discdetsu/med-mlops/releases/download/seed-v2/cxr-seed-bundle-v2.tar.gz
SEED_MODEL_ARGS ?=
ifneq ($(filter http://% https://%,$(SEED_BUNDLE)),)
seed_source := $(SEED_BUNDLE)
seed_mount :=
else
seed_source := /seed/$(notdir $(SEED_BUNDLE))
seed_mount := -v "$(abspath $(dir $(SEED_BUNDLE))):/seed:ro"
endif

.PHONY: help install up down clean ps logs lint fmt typecheck test test-integration test-rules seed train gate promote rollback bootstrap-champion seed-model smoke monitor replay inject-drift label-export demo

help:  ## List targets
	@grep -hE '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-18s %s\n", $$1, $$2}'

.env:
	cp .env.example .env

install:  ## Install the uv workspace and pre-commit hooks
	uv sync --all-packages
	uv run pre-commit install

up: .env  ## Start a profile and wait for health (PROFILE=core|full)
	$(COMPOSE) up -d --build
	scripts/wait_healthy.sh $(PROFILE)

down:  ## Stop containers, keep volumes
	docker compose --profile full down

clean:  ## Stop containers and delete all volumes
	docker compose --profile full down -v

ps:  ## Show service status
	$(COMPOSE) ps -a

logs:  ## Follow logs
	$(COMPOSE) logs -f

lint:  ## ruff check + format check
	uv run ruff check .
	uv run ruff format --check .

fmt:  ## Auto-format and fix lint
	uv run ruff format .
	uv run ruff check --fix .

typecheck:  ## mypy
	uv run mypy

test:  ## Unit tests
	uv run pytest

test-integration:  ## Integration tests against the running core stack
	uv run pytest -m integration

test-rules:  ## Check the Prometheus config and unit-test the alert rules with promtool
	docker run --rm -u 0 -v "$(CURDIR)/compose/prometheus:/etc/prometheus:ro" -w /etc/prometheus/tests \
		--entrypoint sh $(PROMETHEUS_IMAGE) -c \
		'promtool check config /etc/prometheus/prometheus.yml && promtool test rules rules_test.yml'

seed:  ## Download the NIH subset → validate → DVC add + push → Postgres (SEED_ARGS=...)
	uv run python -m cxr_flows.ingest $(SEED_ARGS)

# Django runs the promotion (ADR 0007), so these go through the backend image like seed-model.
backend_cmd := $(COMPOSE) run --rm --no-deps backend python -m cxr_backend

train:  ## Train → evaluate → ONNX → parity → reference → @challenger → gate (CONFIG=demo|full)
	uv run python -m cxr_flows.train --config $(CONFIG) $(TRAIN_ARGS)
	@# Approves @challenger, the version just trained, only if gate.yaml has auto_promote on
	@# and the gate passed it; otherwise a no-op.
	$(backend_cmd) promote --auto

gate:  ## Gate @challenger (or VERSION=N) against @champion; the result waits in Django admin
	uv run python -m cxr_flows.gate $(if $(VERSION),--version $(VERSION)) $(GATE_ARGS)

promote:  ## Approve a gated candidate without the admin (VERSION=N REASON="...")
	@test -n "$(VERSION)" -a -n "$(REASON)" || { echo 'usage: make promote VERSION=N REASON="..."'; exit 2; }
	$(backend_cmd) promote $(VERSION) --reason "$(REASON)"

rollback:  ## Move @champion back to @previous (REASON="...")
	@test -n "$(REASON)" || { echo 'usage: make rollback REASON="..."'; exit 2; }
	$(backend_cmd) rollback --reason "$(REASON)"

bootstrap-champion:  ## Make the first model @champion and build the seed bundle in dist/
	uv run python scripts/bootstrap_champion.py

seed-model:  ## Register the seed bundle as @champion; workers load it (SEED_BUNDLE=URL|path)
	$(COMPOSE) run --rm --no-deps $(seed_mount) backend \
		python -m cxr_backend seed_model $(seed_source) $(SEED_MODEL_ARGS)

smoke:  ## Upload an image through nginx and check the prediction (stack up and seeded)
	scripts/smoke_core.sh

monitor:  ## One monitor pass: drift + delayed AUROC per version (PROFILE=full pushes to Prometheus)
	uv run python -m cxr_flows.monitor run $(MONITOR_ARGS)

replay:  ## Send validation images through the API (ARGS="--n 300 --feedback 0.5")
	uv run python scripts/replay_traffic.py $(ARGS)

inject-drift:  ## Send AP views only, a slice that differs from the reference (ARGS="--n 200")
	uv run python scripts/inject_drift.py $(ARGS)

# Label Studio is in `full`, so these two always run with its environment.
ifeq ($(PROFILE),full)
label-export:  ## Annotations → data/curated → new DVC rev → should_retrain()
	uv run python -m cxr_flows.label_export $(LABEL_EXPORT_ARGS)

demo:  ## The whole loop with no clicks: drift → labels → new rev → retrain → gate → promote
	uv run python scripts/demo.py $(DEMO_ARGS)
else
label-export demo:
	@$(MAKE) --no-print-directory $@ PROFILE=full
endif
