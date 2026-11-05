"""The labeling loop's records (ADR 0009).

The monitor flow inserts a `LabelingTask` for every case it sends to Label Studio, and the export
flow moves it to `exported` or `quarantined` and adds a `RetrainDecision` whenever the outcome of
`should_retrain()` changes. Django owns the tables, as it does `drift_reports` (ADR 0008), so the
admin lists them before the first pass; the flows write them with SQL.
"""

from django.db import models
from django.db.models.functions import Now

from cxr_backend.jobs.models import InferenceJob
from cxr_core import registry


class LabelingTask(models.Model):
    class Status(models.TextChoices):
        OPEN = registry.TASK_OPEN
        EXPORTED = registry.TASK_EXPORTED
        QUARANTINED = registry.TASK_QUARANTINED

    class Reason(models.TextChoices):
        DISAGREEMENT = registry.REASON_DISAGREEMENT
        DRIFT = registry.REASON_DRIFT
        RANDOM = registry.REASON_RANDOM

    # One task per file, like the drift windows: a re-upload is the same observation.
    sha256 = models.CharField(max_length=64, unique=True)
    job = models.ForeignKey(InferenceJob, on_delete=models.CASCADE, related_name="labeling_tasks")
    model_name = models.CharField(max_length=128)
    model_version = models.CharField(max_length=32, db_index=True)
    reason = models.CharField(max_length=16, choices=Reason.choices)
    ls_task_id = models.BigIntegerField(unique=True, help_text="Task ID in Label Studio")
    status = models.CharField(
        max_length=16, choices=Status.choices, db_default=registry.TASK_OPEN, db_index=True
    )
    labels = models.JSONField(null=True, blank=True, help_text="{class: 0 or 1}, the last answer")
    image_id = models.CharField(max_length=64, blank=True, db_default="")
    dataset_rev = models.CharField(
        max_length=64, blank=True, db_default="", help_text="The rev that last changed this row"
    )
    quarantine_reason = models.TextField(blank=True, db_default="")
    created_at = models.DateTimeField(db_default=Now(), db_index=True)
    updated_at = models.DateTimeField(db_default=Now())

    class Meta:
        db_table = registry.LABELING_TASKS_TABLE
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.reason} task {self.ls_task_id}: {self.status}"


class RetrainDecision(models.Model):
    dataset_rev = models.CharField(max_length=64, help_text="The current data")
    baseline_rev = models.CharField(
        max_length=64, blank=True, help_text="The data of the newest model version"
    )
    new_labels = models.PositiveIntegerField(help_text="Curated training images since the baseline")
    min_new_labels = models.PositiveIntegerField(help_text="N in labeling.yaml")
    triggered = models.BooleanField()
    reason = models.TextField()
    train_flow_run_id = models.CharField(max_length=64, blank=True)
    prefect_flow_run_id = models.CharField(max_length=64, blank=True)
    created_at = models.DateTimeField(db_default=Now(), db_index=True)

    class Meta:
        db_table = registry.RETRAIN_DECISIONS_TABLE
        ordering = ["-created_at"]

    def __str__(self) -> str:
        outcome = "retrain" if self.triggered else "no retrain"
        return f"{self.dataset_rev}: {outcome} ({self.new_labels}/{self.min_new_labels})"
