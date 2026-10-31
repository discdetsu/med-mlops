"""Approve, reject, and roll back model versions (ADR 0007). The admin's buttons and the
`promote` / `rollback` management commands both call these, so there is one implementation.

Approve vN: `@previous` → the current champion, `@champion` → vN, `@challenger` dropped if it
pointed at vN; then `model.events` tells every worker to load vN. Rollback: `@champion` →
`@previous`, and `@previous` is cleared, so a second rollback needs another approval first.

Registry changes are serialized with a Postgres advisory lock, so two admins clicking at once
can't interleave alias moves. Each action writes its database rows first and moves the aliases
last, inside the same transaction: a failed alias move rolls the rows back, and a failed commit
puts the aliases back, so MLflow and `model_approvals` agree. Dropping `@challenger`, the
`promotion_status` tag, and telling the workers happen after the commit and are best-effort; if
the publish fails, workers pick the change up when they reconnect or restart (they ask
`/internal/champion` then).
"""

import logging
from collections.abc import Callable, Iterator
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
from cxr_core.registry import (
    CHALLENGER_ALIAS,
    CHAMPION_ALIAS,
    MODEL_NAME,
    PREVIOUS_ALIAS,
    STATUS_APPROVED,
    STATUS_REJECTED,
    STATUS_ROLLED_BACK,
)

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


def rollback_blocker(state: RegistryState) -> str | None:
    """Why `@champion` can't be rolled back right now, or None if it can."""
    if state.champion is None or state.previous is None:
        return (
            f"nothing to roll back to: @{PREVIOUS_ALIAS} is unset (it is set by an approval "
            "and cleared by a rollback)"
        )
    if state.previous == state.champion:
        return (
            f"@{PREVIOUS_ALIAS} is version {state.previous}, which is already @{CHAMPION_ALIAS}; "
            "a rollback would change nothing"
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
    undo: Callable[[], None] | None = None
    try:
        with transaction.atomic(), _registry_lock():
            candidate = ModelCandidate.objects.select_for_update().get(pk=candidate_id)
            champion = alias_version(client, CHAMPION_ALIAS)
            if blocker := approval_blocker(candidate, champion):
                raise PromotionError(
                    f"version {candidate.model_version} can't be approved: {blocker}"
                )
            assert champion is not None  # approval_blocker checked it equals the gated champion
            version = candidate.model_version

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
            # Last, so a failure here rolls the rows above back.
            undo = _move_aliases(client, champion=version, previous=champion)
    except Exception:
        if undo is not None:  # the aliases moved but the commit failed
            undo()
        raise
    _drop_challenger(client, version)
    _tag_status(client, version, STATUS_APPROVED)
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
    _drop_challenger(client, version)
    _tag_status(client, version, STATUS_REJECTED)
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
    undo: Callable[[], None] | None = None
    try:
        with transaction.atomic(), _registry_lock():
            state = registry_state(client)
            if blocker := rollback_blocker(state):
                raise PromotionError(blocker)
            champion, previous = state.champion, state.previous
            assert champion is not None and previous is not None  # rollback_blocker checked

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
            # Last, so a failure here rolls the rows above back.
            undo = _move_aliases(client, champion=previous, previous=None)
    except Exception:
        if undo is not None:  # the aliases moved but the commit failed
            undo()
        raise
    _tag_status(client, champion, STATUS_ROLLED_BACK)
    _announce(client, approval)
    logger.info("rolled back from %s to %s by %s", champion, previous, approval.actor)
    return Decision(approval)


def auto_promote(version: str | None = None, client: MlflowClient | None = None) -> Decision | None:
    """Approve `version` (default: `@challenger`, the version the last train run registered)
    if the gate passed it with `auto_promote` on (gate.yaml) and it is still pending. None if
    it isn't marked, so an older marked candidate is never promoted in its place."""
    client = client or mlflow_client()
    version = version or alias_version(client, CHALLENGER_ALIAS)
    if version is None:
        return None
    candidate = ModelCandidate.objects.filter(
        model_name=MODEL_NAME,
        model_version=version,
        status=ModelCandidate.Status.PENDING,
        gate_passed=True,
        auto_promote=True,
    ).first()
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


def _move_aliases(
    client: MlflowClient, *, champion: str, previous: str | None
) -> Callable[[], None]:
    """Point `@champion` at `champion`, then `@previous` at `previous` (None clears it).

    Either both move or neither does: if the second move fails, the first is put back before
    the error propagates. Returns a function that puts both back, for a caller whose commit
    fails after this succeeded."""
    before = {alias: alias_version(client, alias) for alias in (CHAMPION_ALIAS, PREVIOUS_ALIAS)}

    def undo() -> None:
        for alias, version in before.items():
            try:
                _point_alias(client, alias, version)
            except Exception:
                logger.exception(
                    "couldn't put @%s back to version %s; set it by hand in MLflow", alias, version
                )

    client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, champion)
    try:
        _point_alias(client, PREVIOUS_ALIAS, previous)
    except Exception:
        undo()
        raise
    return undo


def _point_alias(client: MlflowClient, alias: str, version: str | None) -> None:
    if version is not None:
        client.set_registered_model_alias(MODEL_NAME, alias, version)
    elif alias_version(client, alias) is not None:
        client.delete_registered_model_alias(MODEL_NAME, alias)


def _drop_challenger(client: MlflowClient, version: str) -> None:
    """Clear `@challenger` if it still points at `version`; cleanup, so never fatal."""
    try:
        if alias_version(client, CHALLENGER_ALIAS) == version:
            client.delete_registered_model_alias(MODEL_NAME, CHALLENGER_ALIAS)
    except Exception:
        logger.exception("couldn't clear @%s from version %s", CHALLENGER_ALIAS, version)


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
