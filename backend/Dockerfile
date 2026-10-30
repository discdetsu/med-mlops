# Build context: the repo root (uv workspace). See .dockerignore.
FROM python:3.14.9-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /src
# Dependencies first, so a code change doesn't reinstall them. Every member's pyproject.toml
# is needed for uv to read the workspace, even the ones not installed.
COPY pyproject.toml uv.lock .python-version ./
COPY packages/cxr_core/pyproject.toml packages/cxr_core/
COPY backend/pyproject.toml backend/
COPY worker/pyproject.toml worker/
COPY flows/pyproject.toml flows/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package cxr-backend --no-install-workspace
COPY packages/cxr_core/src packages/cxr_core/src
COPY backend/src backend/src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package cxr-backend --no-editable

FROM python:3.14.9-slim
RUN useradd --system --uid 10001 --no-create-home app
COPY --from=build /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1 \
    DJANGO_SETTINGS_MODULE=cxr_backend.settings DJANGO_STATIC_ROOT=/app/static
# The admin's static files, served by WhiteNoise. Settings need these values to import.
RUN DJANGO_SECRET_KEY=build APP_DATABASE_URL=sqlite:// INTERNAL_API_TOKEN=build \
    CELERY_BROKER_URL=memory:// python -m cxr_backend collectstatic --noinput -v 0
USER app
EXPOSE 8000
# One process with threads: Prometheus metrics live in that process's registry, and a
# demo stack doesn't need more. Migrations are idempotent and run on every start.
CMD ["sh", "-c", "python -m cxr_backend migrate --noinput && python -m cxr_backend ensure_admin && exec gunicorn cxr_backend.wsgi --bind 0.0.0.0:8000 --workers 1 --threads 8 --access-logfile -"]
