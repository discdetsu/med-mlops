import os
import urllib.request
from typing import Any

import pytest
from boto3.s3.transfer import TransferConfig

pytestmark = pytest.mark.integration


def test_buckets_exist(s3: Any) -> None:
    names = {b["Name"] for b in s3.list_buckets()["Buckets"]}
    assert {"datasets", "artifacts", "inference", "mlflow"} <= names


def test_inference_raw_lifecycle_rule(s3: Any) -> None:
    rules = s3.get_bucket_lifecycle_configuration(Bucket="inference")["Rules"]
    assert any(r["Filter"].get("Prefix") == "raw/" and r["Status"] == "Enabled" for r in rules)


def test_put_get_roundtrip(s3: Any, scratch_prefix: str) -> None:
    key = scratch_prefix + "hello.txt"
    s3.put_object(Bucket="artifacts", Key=key, Body=b"hello rustfs")
    assert s3.get_object(Bucket="artifacts", Key=key)["Body"].read() == b"hello rustfs"


def test_multipart_upload(s3: Any, scratch_prefix: str, tmp_path: Any) -> None:
    payload = os.urandom(12 * 1024 * 1024)
    src = tmp_path / "big.bin"
    src.write_bytes(payload)
    config = TransferConfig(
        multipart_threshold=5 * 1024 * 1024, multipart_chunksize=5 * 1024 * 1024
    )

    key = scratch_prefix + "big.bin"
    s3.upload_file(str(src), "artifacts", key, Config=config)

    head = s3.head_object(Bucket="artifacts", Key=key)
    assert "-" in head["ETag"], "expected a multipart ETag"
    assert s3.get_object(Bucket="artifacts", Key=key)["Body"].read() == payload


def test_presigned_get_url(s3: Any, scratch_prefix: str) -> None:
    key = scratch_prefix + "presigned.txt"
    s3.put_object(Bucket="artifacts", Key=key, Body=b"signed")
    url = s3.generate_presigned_url(
        "get_object", Params={"Bucket": "artifacts", "Key": key}, ExpiresIn=60
    )
    with urllib.request.urlopen(url) as resp:
        assert resp.read() == b"signed"
