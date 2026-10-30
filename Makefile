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
endif

CONFIG ?= demo
SEED_ARGS ?=
TRAIN_ARGS ?=

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

.PHONY: help install up down clean ps logs lint fmt typecheck test test-integration seed train bootstrap-champion seed-model smoke demo

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

seed:  ## Download the NIH subset → validate → DVC add + push → Postgres (SEED_ARGS=...)
	uv run python -m cxr_flows.ingest $(SEED_ARGS)

train:  ## Train → evaluate → ONNX → parity → reference → register @challenger (CONFIG=demo|full)
	uv run python -m cxr_flows.train --config $(CONFIG) $(TRAIN_ARGS)

bootstrap-champion:  ## Make the first model @champion and build the seed bundle in dist/
	uv run python scripts/bootstrap_champion.py

seed-model:  ## Register the seed bundle as @champion; workers load it (SEED_BUNDLE=URL|path)
	$(COMPOSE) run --rm --no-deps $(seed_mount) backend \
		python -m cxr_backend seed_model $(seed_source) $(SEED_MODEL_ARGS)

smoke:  ## Upload an image through nginx and check the prediction (stack up and seeded)
	scripts/smoke_core.sh

demo:  ## Phase 5: scripted label → retrain → promote loop
	@echo "not implemented yet (Phase 5)"; exit 1
