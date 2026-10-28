"""MLflow registration and bundle publishing.

The registered version's ONNX model is logged with the MLflow ONNX flavor; the serving bundle
(`cxr_core.bundle`) is logged next to it under `bundle/` and copied to
`s3://artifacts/models/<name>/<version>/`, where workers load it from (Phase 2). The
`@challenger` alias moves only after every tag and upload succeeded.
"""

import contextlib
import dataclasses
import json
from collections.abc import Mapping
from pathlib import Path

import boto3
import mlflow
import mlflow.onnx
import onnx
from mlflow import MlflowClient

from cxr_core.bundle import BUNDLE_FILES, Manifest
from cxr_flows.settings import ARTIFACTS_BUCKET, CHALLENGER_ALIAS


def bundle_prefix(model_name: str, version: str) -> str:
    return f"models/{model_name}/{version}/"


def register(
    run_id: str,
    bundle_dir: Path,
    draft: Manifest,
    tags: Mapping[str, str],
) -> str:
    """Register the bundle's ONNX model as a new version and return the version number.

    `draft` is the bundle manifest with a placeholder `model_version`; the real one is only
    known once MLflow assigns it, and is written into `bundle_dir` before uploading.
    """
    model_name = draft.model_name
    client = MlflowClient()
    active = mlflow.active_run()
    run_ctx = (
        contextlib.nullcontext()
        if active is not None and active.info.run_id == run_id
        else mlflow.start_run(run_id=run_id)
    )
    with run_ctx:
        info = mlflow.onnx.log_model(
            onnx.load(str(bundle_dir / "model.onnx")),
            name="model",
            registered_model_name=model_name,
            save_as_external_data=False,
            # Serving runs the bundle with ONNX Runtime, not this MLflow flavor; skip the slow
            # export of the whole uv lockfile as the model's environment.
            pip_requirements=["onnx", "onnxruntime"],
        )
    version = str(info.registered_model_version)

    manifest = dataclasses.replace(draft, model_version=version)
    manifest.save(bundle_dir)
    client.log_artifacts(run_id, str(bundle_dir), artifact_path="bundle")

    prefix = bundle_prefix(model_name, version)
    s3 = boto3.client("s3")
    for name in BUNDLE_FILES:
        s3.upload_file(str(bundle_dir / name), ARTIFACTS_BUCKET, prefix + name)

    all_tags = {
        **tags,
        "thresholds": json.dumps(manifest.thresholds, sort_keys=True),
        "bundle_uri": f"s3://{ARTIFACTS_BUCKET}/{prefix}",
        "mlflow_run_id": run_id,
    }
    for key, value in all_tags.items():
        client.set_model_version_tag(model_name, version, key, value)
    client.set_registered_model_alias(model_name, CHALLENGER_ALIAS, version)
    return version
