# Build context: the repo root (uv workspace). See .dockerignore. No Django, no torch:
# cxr-core without its [train] extra, ONNX Runtime, Celery.
FROM python:3.14.9-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /src
COPY pyproject.toml uv.lock .python-version ./
COPY packages/cxr_core/pyproject.toml packages/cxr_core/
COPY backend/pyproject.toml backend/
COPY worker/pyproject.toml worker/
COPY flows/pyproject.toml flows/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package cxr-worker --no-install-workspace
COPY packages/cxr_core/src packages/cxr_core/src
COPY worker/src worker/src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package cxr-worker --no-editable

FROM python:3.14.9-slim
RUN useradd --system --uid 10001 --no-create-home app
COPY --from=build /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1 MODEL_CACHE_DIR=/tmp/cxr-models
USER app
# The pool is set to solo in cxr_worker.app; see its docstring for why.
CMD ["celery", "-A", "cxr_worker.app", "worker", "-Q", "inference", "--loglevel", "INFO", \
     "--without-gossip", "--without-mingle"]
