"""Integration tests run on the host against `make up` (core profile).

`make test-integration` exports the host-side env from .env; running pytest directly
falls back to reading .env here, so both paths hit the same endpoints.
"""

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import boto3
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv() -> dict[str, str]:
    path = REPO_ROOT / ".env"
    if not path.exists():
        path = REPO_ROOT / ".env.example"
    values = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


@pytest.fixture(scope="session", autouse=True)
def host_env() -> None:
    env = _load_dotenv()
    os.environ.setdefault("AWS_ENDPOINT_URL", env["S3_HOST_ENDPOINT_URL"])
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", env["S3_HOST_ENDPOINT_URL"])
    os.environ.setdefault("AWS_ACCESS_KEY_ID", env["S3_ACCESS_KEY"])
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", env["S3_SECRET_KEY"])
    os.environ.setdefault("AWS_DEFAULT_REGION", env["S3_REGION"])
    os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")
    # See docs/adr/0001: artifacts stream through MLflow, never presigned to rustfs:9000.
    os.environ.setdefault("MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD", "false")
    os.environ.setdefault("MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD", "false")
    os.environ.setdefault("RABBITMQ_URL", _rabbitmq_url(env))
    os.environ.setdefault(
        "APP_DATABASE_URL", f"postgresql://app:{env['APP_DB_PASSWORD']}@localhost:5432/app"
    )


def _rabbitmq_url(env: dict[str, str]) -> str:
    return f"amqp://{env['RABBITMQ_USER']}:{env['RABBITMQ_PASSWORD']}@localhost:5672//"


@pytest.fixture(scope="session")
def s3(host_env: None) -> Any:
    # No endpoint_url argument: boto3 must pick it up from AWS_ENDPOINT_URL like every
    # other client does.
    return boto3.client("s3")


@pytest.fixture
def scratch_prefix(s3: Any, request: pytest.FixtureRequest) -> Iterator[str]:
    """A per-test key prefix in the `artifacts` bucket, deleted afterwards."""
    prefix = f"_integration/{request.node.name}/"
    yield prefix
    listed = s3.list_objects_v2(Bucket="artifacts", Prefix=prefix)
    for obj in listed.get("Contents", []):
        s3.delete_object(Bucket="artifacts", Key=obj["Key"])
