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


def aliases(client: MlflowClient) -> dict[str, str]:
    """Alias → version for the model; empty before it is first registered.

    Read from the registered model rather than by alias: for a model that exists, MLflow
    answers a missing alias with INVALID_PARAMETER_VALUE, which can't be told apart from a
    real error without parsing the message.
    """
    try:
        model = client.get_registered_model(MODEL_NAME)
    except MlflowException as exc:
        if exc.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST):
            return {}
        raise
    return {alias: str(version) for alias, version in model.aliases.items()}


def alias_version(client: MlflowClient, alias: str) -> str | None:
    """The version holding `alias`, or None when none does (or the model doesn't exist yet)."""
    return aliases(client).get(alias)


def current_champion(client: MlflowClient | None = None) -> Champion | None:
    """None when no version holds the alias yet (before `make seed-model`)."""
    client = client or mlflow_client()
    version = alias_version(client, CHAMPION_ALIAS)
    if version is None:
        return None
    mv = client.get_model_version(MODEL_NAME, version)
    return Champion(
        model_name=MODEL_NAME,
        model_version=version,
        bundle_uri=mv.tags.get("bundle_uri") or bundle_uri(MODEL_NAME, version),
        mlflow_run_id=mv.run_id or "",
    )
