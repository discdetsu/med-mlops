"""The current `@champion`, read from the MLflow registry (the source of truth for aliases)."""

from dataclasses import dataclass

from django.conf import settings
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST, ErrorCode

from cxr_core.registry import CHAMPION_ALIAS, MODEL_NAME, bundle_uri


@dataclass(frozen=True)
class Champion:
    model_name: str
    model_version: str
    bundle_uri: str
    mlflow_run_id: str


def mlflow_client() -> MlflowClient:
    return MlflowClient(tracking_uri=settings.MLFLOW_TRACKING_URI)


def current_champion(client: MlflowClient | None = None) -> Champion | None:
    """None when no version holds the alias yet (before `make seed-model`)."""
    client = client or mlflow_client()
    try:
        mv = client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    except MlflowException as exc:
        if exc.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST):
            return None
        raise
    version = str(mv.version)
    return Champion(
        model_name=MODEL_NAME,
        model_version=version,
        bundle_uri=mv.tags.get("bundle_uri") or bundle_uri(MODEL_NAME, version),
        mlflow_run_id=mv.run_id or "",
    )
