import logging

from django.http import HttpRequest
from mlflow.exceptions import MlflowException
from ninja import Router, Schema, Status

from cxr_backend.jobs.schemas import Error
from cxr_backend.registry import promotion
from cxr_backend.registry.champion import current_champion

logger = logging.getLogger(__name__)

internal_router = Router(tags=["internal"])


class ChampionOut(Schema):
    model_name: str
    model_version: str
    bundle_uri: str
    mlflow_run_id: str


class AutoPromoteIn(Schema):
    version: str


class AutoPromoteOut(Schema):
    promoted: bool
    from_version: str = ""
    to_version: str = ""
    event_published: bool = False


@internal_router.post("/auto-promote", response={200: AutoPromoteOut, 409: Error, 503: Error})
def post_auto_promote(
    request: HttpRequest, payload: AutoPromoteIn
) -> Status[AutoPromoteOut | Error]:
    """`promote --auto` for a train run started from a Prefect deployment, which has no Django of
    its own: approves the version only if the gate marked it, and is a no-op otherwise."""
    try:
        decision = promotion.auto_promote(payload.version)
    except promotion.PromotionError as exc:
        return Status(409, Error(detail=str(exc)))
    except MlflowException:
        logger.exception("auto-promote of version %s failed", payload.version)
        return Status(503, Error(detail="the model registry is unavailable"))
    if decision is None:
        return Status(200, AutoPromoteOut(promoted=False))
    approval = decision.approval
    return Status(
        200,
        AutoPromoteOut(
            promoted=True,
            from_version=approval.from_version,
            to_version=approval.to_version,
            event_published=decision.event_published,
        ),
    )


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
