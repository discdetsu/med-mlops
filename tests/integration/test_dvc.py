import subprocess
import uuid
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.integration


def _run(cwd: Path, *args: str) -> None:
    subprocess.run(args, cwd=cwd, check=True, capture_output=True)


def test_push_pull_roundtrip(s3: Any, tmp_path: Path) -> None:
    """DVC reaches RustFS through AWS_ENDPOINT_URL alone, with no endpointurl in config."""
    prefix = f"_integration/dvc-{uuid.uuid4().hex}"
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(repo, "git", "init", "-q")
    _run(repo, "dvc", "init", "-q")
    _run(repo, "dvc", "remote", "add", "-d", "test", f"s3://datasets/{prefix}")

    data = repo / "sample.bin"
    payload = uuid.uuid4().bytes * 1024
    data.write_bytes(payload)
    _run(repo, "dvc", "add", "sample.bin")
    _run(repo, "dvc", "push")

    data.unlink()
    _run(repo, "dvc", "gc", "--workspace", "--force")  # empty the local cache too
    _run(repo, "dvc", "pull")
    assert data.read_bytes() == payload

    for obj in s3.list_objects_v2(Bucket="datasets", Prefix=prefix).get("Contents", []):
        s3.delete_object(Bucket="datasets", Key=obj["Key"])
