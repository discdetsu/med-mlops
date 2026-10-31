"""Promotion records (ADR 0007).

`ModelCandidate`: one row per gated version, inserted by the gate flow (cxr_flows.db) with the
gate's report, then moved by the admin's approve / reject / rollback actions. Django owns the
table so it exists, and the admin works, before any gate has run.

`ModelApproval`: who decided what, when, and why. Written only by `registry.promotion`.

The MLflow aliases stay the source of truth for which version serves; these rows are the
audit trail and the queue of decisions.
"""

from django.conf import settings
from django.db import models
from django.db.models.functions import Now

from cxr_core import registry


class ModelCandidate(models.Model):
    class Status(models.TextChoices):
        PENDING = registry.STATUS_PENDING, "pending approval"
        GATE_FAILED = registry.STATUS_GATE_FAILED, "gate failed"
        APPROVED = registry.STATUS_APPROVED, "approved"
        REJECTED = registry.STATUS_REJECTED, "rejected"
        ROLLED_BACK = registry.STATUS_ROLLED_BACK, "rolled back"

    model_name = models.CharField(max_length=128)
    model_version = models.CharField(max_length=32)
    champion_version = models.CharField(
        max_length=32, help_text="The @champion the gate compared this version with"
    )
    status = models.CharField(max_length=16, choices=Status.choices, db_index=True)
    gate_passed = models.BooleanField()
    auto_promote = models.BooleanField(
        db_default=False, help_text="Set by the gate when gate.yaml has auto_promote on"
    )
    report = models.JSONField(help_text="The gate report (cxr_flows.gating.decide)")
    report_uri = models.CharField(max_length=512, help_text="HTML gate report in object storage")
    mlflow_run_id = models.CharField(max_length=64, blank=True)
    dataset_rev = models.CharField(max_length=64, blank=True)
    gated_at = models.DateTimeField(db_default=Now())
    decided_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = registry.CANDIDATES_TABLE
        ordering = ["-gated_at"]
        constraints = [
            # The gate flow upserts on these columns.
            models.UniqueConstraint(
                fields=["model_name", "model_version"], name="model_candidates_version_unique"
            )
        ]
        permissions = [("promote_model", "Can approve, reject, and roll back model versions")]

    def __str__(self) -> str:
        return f"{self.model_name} v{self.model_version} ({self.get_status_display()})"


class ModelApproval(models.Model):
    class Action(models.TextChoices):
        APPROVE = "approve"
        REJECT = "reject"
        ROLLBACK = "rollback"

    candidate = models.ForeignKey(
        ModelCandidate,
        null=True,
        blank=True,
        on_delete=models.PROTECT,
        related_name="decisions",
        help_text="The approved or rejected candidate; for a rollback, the one rolled back",
    )
    action = models.CharField(max_length=16, choices=Action.choices)
    model_name = models.CharField(max_length=128)
    from_version = models.CharField(
        max_length=32, blank=True, help_text="@champion before the action"
    )
    to_version = models.CharField(max_length=32, blank=True, help_text="@champion after the action")
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL
    )
    actor = models.CharField(
        max_length=150, help_text="Username at the time, or `auto_promote` / `cli`"
    )
    reason = models.TextField()
    event_published = models.BooleanField(
        default=False, help_text="Whether workers were told about the new @champion"
    )
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        db_table = "model_approvals"
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.action} by {self.actor}: v{self.from_version} → v{self.to_version}"
