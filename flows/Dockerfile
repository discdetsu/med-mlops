# The `monitor` service (full profile): the flows package without torch. Build context: the
# repo root (uv workspace), like backend/ and worker/.
#
# cxr-flows depends on cxr-core[train] for training; the monitor flow never imports torch, so
# torch, torchvision, and timm are left out of this image (about 1.5 GB). Only flows that don't
# train or export run here; Phase 5's train deployment needs an image with them.
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
ARG SKIP="--no-install-package torch --no-install-package torchvision --no-install-package timm"
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package cxr-flows --no-install-workspace $SKIP
COPY packages/cxr_core/src packages/cxr_core/src
COPY flows/src flows/src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package cxr-flows --no-editable $SKIP

FROM python:3.14.9-slim
# The reference cache volume mounts here; a new named volume takes this directory's owner.
RUN useradd --system --uid 10001 --create-home app \
    && mkdir -p /home/app/.cache && chown app:app /home/app/.cache
COPY --from=build /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1 MONITOR_CACHE_DIR=/home/app/.cache \
    MLFLOW_DISABLE_AGENT_HINT=1
USER app
WORKDIR /home/app
# Fails at build time if the monitor flow ever starts importing torch.
RUN python -c "import cxr_flows.monitor, sys; assert 'torch' not in sys.modules"
CMD ["python", "-m", "cxr_flows.monitor", "serve"]
