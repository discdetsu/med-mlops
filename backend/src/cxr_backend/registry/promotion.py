"""Approve, reject, and roll back model versions (ADR 0007). The admin's buttons and the
`promote` / `rollback` management commands both call these, so there is one implementation.

Approve vN: `@previous` → the current champion, `@champion` → vN, `@challenger` dropped if it
pointed at vN; then `model.events` tells every worker to load vN. Rollback: `@champion` →
`@previous`, and `@previous` is cleared, so a second rollback needs another approval first.

Registry changes are serialized with a Postgres advisory lock, so two admins clicking at once
can't interleave alias moves. Workers are told only after the database commit; if that publish
fails, the aliases have still moved, and workers pick the change up when they reconnect or
restart (they ask `/internal/champion` then).
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from django.contrib.auth.models import User
from django.db import connection, transaction
from django.utils import timezone
from mlflow import MlflowClient

from cxr_backend.registry.champion import (
    alias_version,
    aliases,
    current_champion,
    mlflow_client,
)
from cxr_backend.registry.events import publish_champion_changed
from cxr_backend.registry.models import ModelApproval, ModelCandidate
from cxr_core.messaging import ChampionChanged
from cxr_core.registry import CHALLENGER_ALIAS, CHAMPION_ALIAS, MODEL_NAME, PREVIOUS_ALIAS

logger = logging.getLogger(__name__)

# Any constant shared by every backend process; the value only has to be unique in the app DB.
_REGISTRY_LOCK_ID = 0x63787270  # "cxrp"
AUTO_PROMOTE_ACTOR = "auto_promote"


class PromotionError(Exception):
    """The action isn't allowed in the current state; nothing was changed."""


@dataclass(frozen=True)
class RegistryState:
    champion: str | None
    previous: str | None
    challenger: str | None


@dataclass(frozen=True)
class Decision:
    approval: ModelApproval

    @property
    def event_published(self) -> bool:
        return self.approval.event_published


def registry_state(client: MlflowClient | None = None) -> RegistryState:
    current = aliases(client or mlflow_client())
    return RegistryState(
        champion=current.get(CHAMPION_ALIAS),
        previous=current.get(PREVIOUS_ALIAS),
        challenger=current.get(CHALLENGER_ALIAS),
    )


def approval_blocker(candidate: ModelCandidate, champion: str | None) -> str | None:
    """Why `candidate` can't be approved right now, or None if it can."""
    if candidate.status != ModelCandidate.Status.PENDING:
        return f"it is {candidate.get_status_display()}, not pending approval"
    if not candidate.gate_passed:
        return "it did not pass the gate"
    if champion != candidate.champion_version:
        return (
            f"the gate compared it with version {candidate.champion_version}, but @champion is "
            f"now {champion or 'unset'}; re-run the gate: make gate VERSION="
            f"{candidate.model_version}"
        )
    return None


def approve(
    candidate_id: int,
    *,
    reason: str,
    user: User | None = None,
    actor: str | None = None,
    client: MlflowClient | None = None,
) -> Decision:
    reason = _require_reason(reason)
    client = client or mlflow_client()
    with transaction.atomic(), _registry_lock():
        candidate = ModelCandidate.objects.select_for_update().get(pk=candidate_id)
        champion = alias_version(client, CHAMPION_ALIAS)
        if blocker := approval_blocker(candidate, champion):
            raise PromotionError(f"version {candidate.model_version} can't be approved: {blocker}")
        assert champion is not None  # approval_blocker checked it equals the gated champion
        version = candidate.model_version

        client.set_registered_model_alias(MODEL_NAME, PREVIOUS_ALIAS, champion)
        if alias_version(client, CHALLENGER_ALIAS) == version:
            client.delete_registered_model_alias(MODEL_NAME, CHALLENGER_ALIAS)
        # The switch, and the last MLflow call allowed to fail: only database writes follow.
        client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, version)
        _tag_status(client, version, "approved")

        candidate.status = ModelCandidate.Status.APPROVED
        candidate.decided_at = timezone.now()
        candidate.save(update_fields=["status", "decided_at"])
        approval = ModelApproval.objects.create(
            candidate=candidate,
            action=ModelApproval.Action.APPROVE,
            model_name=candidate.model_name,
            from_version=champion,
            to_version=version,
            user=user,
            actor=_actor(user, actor),
            reason=reason,
        )
    _announce(client, approval)
    logger.info("version %s approved by %s; was %s", version, approval.actor, champion)
    return Decision(approval)


def reject(
    candidate_id: int,
    *,
    reason: str,
    user: User | None = None,
    actor: str | None = None,
    client: MlflowClient | None = None,
) -> Decision:
    """Close a pending or gate-failed candidate. `@challenger` is dropped if it points at it."""
    reason = _require_reason(reason)
    client = client or mlflow_client()
    open_statuses = (ModelCandidate.Status.PENDING, ModelCandidate.Status.GATE_FAILED)
    with transaction.atomic(), _registry_lock():
        candidate = ModelCandidate.objects.select_for_update().get(pk=candidate_id)
        if candidate.status not in open_statuses:
            raise PromotionError(
                f"version {candidate.model_version} is {candidate.get_status_display()}; only "
                "pending or gate-failed candidates can be rejected"
            )
        version = candidate.model_version
        champion = alias_version(client, CHAMPION_ALIAS) or ""
        if alias_version(client, CHALLENGER_ALIAS) == version:
            client.delete_registered_model_alias(MODEL_NAME, CHALLENGER_ALIAS)
        _tag_status(client, version, "rejected")

        candidate.status = ModelCandidate.Status.REJECTED
        candidate.decided_at = timezone.now()
        candidate.save(update_fields=["status", "decided_at"])
        approval = ModelApproval.objects.create(
            candidate=candidate,
            action=ModelApproval.Action.REJECT,
            model_name=candidate.model_name,
            from_version=champion,
            to_version=champion,
            user=user,
            actor=_actor(user, actor),
            reason=reason,
        )
    logger.info("version %s rejected by %s", version, approval.actor)
    return Decision(approval)


def rollback(
    *,
    reason: str,
    user: User | None = None,
    actor: str | None = None,
    client: MlflowClient | None = None,
) -> Decision:
    """`@champion` back to `@previous`; the rolled-back version's candidate is marked so."""
    reason = _require_reason(reason)
    client = client or mlflow_client()
    with transaction.atomic(), _registry_lock():
        champion = alias_version(client, CHAMPION_ALIAS)
        previous = alias_version(client, PREVIOUS_ALIAS)
        if champion is None or previous is None:
            raise PromotionError(
                f"nothing to roll back to: @{PREVIOUS_ALIAS} is unset (it is set by an approval "
                "and cleared by a rollback)"
            )
        # The switch, and the last MLflow call allowed to fail: only database writes follow.
        client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, previous)
        try:
            client.delete_registered_model_alias(MODEL_NAME, PREVIOUS_ALIAS)
        except Exception:
            # @previous now equals @champion, so another rollback would be a no-op swap.
            logger.exception("couldn't clear @%s after rolling back", PREVIOUS_ALIAS)
        _tag_status(client, champion, "rolled_back")

        candidate = (
            ModelCandidate.objects.select_for_update()
            .filter(model_name=MODEL_NAME, model_version=champion)
            .first()
        )
        if candidate is not None:
            candidate.status = ModelCandidate.Status.ROLLED_BACK
            candidate.decided_at = timezone.now()
            candidate.save(update_fields=["status", "decided_at"])
        approval = ModelApproval.objects.create(
            candidate=candidate,
            action=ModelApproval.Action.ROLLBACK,
            model_name=MODEL_NAME,
            from_version=champion,
            to_version=previous,
            user=user,
            actor=_actor(user, actor),
            reason=reason,
        )
    _announce(client, approval)
    logger.info("rolled back from %s to %s by %s", champion, previous, approval.actor)
    return Decision(approval)


def auto_promote(client: MlflowClient | None = None) -> Decision | None:
    """Approve the newest candidate the gate passed with `auto_promote` on (gate.yaml), if it
    is still pending and was gated against the current champion. None if there is none."""
    candidate = (
        ModelCandidate.objects.filter(
            status=ModelCandidate.Status.PENDING, gate_passed=True, auto_promote=True
        )
        .order_by("-gated_at")
        .first()
    )
    if candidate is None:
        return None
    return approve(
        candidate.pk,
        reason="auto_promote is on in gate.yaml and the gate passed",
        actor=AUTO_PROMOTE_ACTOR,
        client=client,
    )


def _announce(client: MlflowClient, approval: ModelApproval) -> None:
    """Tell every worker about the new @champion; record whether that worked."""
    try:
        champion = current_champion(client)
        if champion is None or champion.model_version != approval.to_version:
            raise RuntimeError(f"@champion is {champion}, expected version {approval.to_version}")
        publish_champion_changed(
            ChampionChanged(champion.model_name, champion.model_version, champion.bundle_uri)
        )
    except Exception:
        logger.exception(
            "@champion is version %s but workers weren't told; they load it when they "
            "reconnect or restart",
            approval.to_version,
        )
        return
    approval.event_published = True
    approval.save(update_fields=["event_published"])


def _tag_status(client: MlflowClient, version: str, status: str) -> None:
    """Mirror the decision on the MLflow version for lineage; informational, so never fatal."""
    try:
        client.set_model_version_tag(MODEL_NAME, version, "promotion_status", status)
    except Exception:
        logger.exception("couldn't tag version %s as %s", version, status)


@contextmanager
def _registry_lock() -> Iterator[None]:
    """Serialize registry changes across backend processes until the transaction ends."""
    if connection.vendor == "postgresql":
        with connection.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(%s)", [_REGISTRY_LOCK_ID])
    yield


def _require_reason(reason: str) -> str:
    reason = reason.strip()
    if not reason:
        raise PromotionError("a reason is required")
    return reason


def _actor(user: User | None, actor: str | None) -> str:
    if actor:
        return actor
    if user is not None:
        return user.get_username()
    return "cli"
