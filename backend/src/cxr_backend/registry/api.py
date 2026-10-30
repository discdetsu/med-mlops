import logging

from django.http import HttpRequest
from mlflow.exceptions import MlflowException
from ninja import Router, Schema, Status

from cxr_backend.jobs.schemas import Error
from cxr_backend.registry.champion import current_champion

logger = logging.getLogger(__name__)

internal_router = Router(tags=["internal"])


class ChampionOut(Schema):
    model_name: str
    model_version: str
    bundle_uri: str
    mlflow_run_id: str


@internal_router.get("/champion", response={200: ChampionOut, 404: Error, 503: Error})
def get_champion(request: HttpRequest) -> Status[ChampionOut | Error]:
    """Workers call this on startup and after reconnecting, then load the bundle."""
    try:
        champion = current_champion()
    except MlflowException:
        logger.exception("champion lookup failed")
        return Status(503, Error(detail="the model registry is unavailable"))
    if champion is None:
        return Status(404, Error(detail="no @champion yet; run `make seed-model`"))
    return Status(
        200,
        ChampionOut(
            model_name=champion.model_name,
            model_version=champion.model_version,
            bundle_uri=champion.bundle_uri,
            mlflow_run_id=champion.mlflow_run_id,
        ),
    )
