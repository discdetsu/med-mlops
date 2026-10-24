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

.PHONY: help install up down clean ps logs lint fmt typecheck test test-integration seed seed-model train demo

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

seed:  ## Phase 1: download the NIH subset and version it with DVC
	@echo "not implemented yet (Phase 1)"; exit 1

train:  ## Phase 1: run the train flow
	@echo "not implemented yet (Phase 1)"; exit 1

seed-model:  ## Phase 2: download the seed bundle and register it as @champion
	@echo "not implemented yet (Phase 2)"; exit 1

demo:  ## Phase 5: scripted label → retrain → promote loop
	@echo "not implemented yet (Phase 5)"; exit 1
