"""S3 access for the backend. The endpoint and credentials come from the AWS_* env vars that
every client shares (ADR 0001), so there is no endpoint argument here."""

from functools import cache
from typing import IO, Any

import boto3
from django.conf import settings


@cache
def _client() -> Any:
    return boto3.client("s3")


def put_raw(key: str, body: IO[bytes], content_type: str) -> None:
    _client().upload_fileobj(
        body, settings.INFERENCE_BUCKET, key, ExtraArgs={"ContentType": content_type}
    )


def get_inference_object(key: str) -> bytes:
    response = _client().get_object(Bucket=settings.INFERENCE_BUCKET, Key=key)
    body: bytes = response["Body"].read()
    return body


def upload_file(path: str, bucket: str, key: str) -> None:
    _client().upload_file(path, bucket, key)


def get_object(bucket: str, key: str) -> bytes:
    response = _client().get_object(Bucket=bucket, Key=key)
    body: bytes = response["Body"].read()
    return body
