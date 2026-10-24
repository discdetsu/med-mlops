import uuid
from pathlib import Path
from typing import Any

import mlflow
import pytest

pytestmark = pytest.mark.integration


def test_artifact_lands_in_rustfs(s3: Any, tmp_path: Path) -> None:
    mlflow.set_experiment("_integration")
    marker = uuid.uuid4().hex
    artifact = tmp_path / "marker.txt"
    artifact.write_text(marker)

    with mlflow.start_run() as run:
        mlflow.log_artifact(str(artifact))

    # Proxied artifact upload: the client talks to MLflow, MLflow writes to s3://mlflow.
    downloaded = mlflow.artifacts.download_artifacts(
        run_id=run.info.run_id, artifact_path="marker.txt", dst_path=str(tmp_path / "dl")
    )
    assert Path(downloaded).read_text() == marker

    key = f"{run.info.experiment_id}/{run.info.run_id}/artifacts/marker.txt"
    body = s3.get_object(Bucket="mlflow", Key=key)["Body"].read()
    assert body.decode() == marker
