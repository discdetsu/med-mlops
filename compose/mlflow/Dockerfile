FROM python:3.14.9-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /bin/uv
RUN uv pip install --system --no-cache \
        "mlflow==3.17.2" \
        "psycopg2-binary>=2.9,<3" \
        "boto3>=1.43,<2"

EXPOSE 5000
