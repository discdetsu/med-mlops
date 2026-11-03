"""Drift reports (ADR 0008).

The monitor flow (`cxr_flows.monitor`) inserts one row per window it compares with a model
version's drift reference, and puts the Evidently HTML report in object storage. Django owns
the table, as it does `model_candidates` (ADR 0007), so the admin lists reports before the
first monitor run instead of erroring on a missing table.

The scores are also pushed to Prometheus through the Pushgateway, where the alerts fire; these
rows are the history and the way to the full report.
"""

from django.db import models
from django.db.models.functions import Now

from cxr_core import registry


class DriftReport(models.Model):
    model_name = models.CharField(max_length=128)
    model_version = models.CharField(max_length=32, db_index=True)
    window_start = models.DateTimeField(help_text="Oldest prediction in the window")
    window_end = models.DateTimeField(help_text="Newest prediction in the window")
    n_predictions = models.PositiveIntegerField()
    n_reference = models.PositiveIntegerField(help_text="Reference rows compared against")
    window_fingerprint = models.CharField(
        max_length=64, help_text="Hash of the window's job ids; the flow skips unchanged windows"
    )
    drift_detected = models.BooleanField(help_text="Any score above its threshold")
    embedding_drift = models.FloatField(null=True, blank=True)
    prediction_drift = models.FloatField(null=True, blank=True, help_text="Largest per-class score")
    metadata_drift = models.FloatField(
        null=True, blank=True, help_text="Largest score over sex, age, and view"
    )
    report = models.JSONField(help_text="Scores and thresholds per feature, delayed AUROC")
    report_uri = models.CharField(max_length=512, help_text="Evidently HTML report")
    prefect_flow_run_id = models.CharField(max_length=64, blank=True)
    created_at = models.DateTimeField(db_default=Now(), db_index=True)

    class Meta:
        db_table = registry.DRIFT_REPORTS_TABLE
        ordering = ["-created_at"]

    def __str__(self) -> str:
        state = "drift" if self.drift_detected else "no drift"
        return f"{self.model_name} v{self.model_version}, {self.n_predictions} predictions: {state}"
