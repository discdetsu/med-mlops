"""Reviewer feedback on a prediction: agree, or the corrected labels.

`labels` always holds the reviewer's full answer per class (for "agree", the predicted
positives), so the monitor flow can compute delayed AUROC per `model_version` (Phase 4) and
find disagreements to send to Label Studio (Phase 5) from this table alone.
"""

from django.db import models

from cxr_backend.jobs.models import InferenceJob


class Feedback(models.Model):
    class Verdict(models.TextChoices):
        AGREE = "agree"
        CORRECT = "correct"

    job = models.ForeignKey(InferenceJob, on_delete=models.CASCADE, related_name="feedback")
    model_version = models.CharField(max_length=32, db_index=True)
    verdict = models.CharField(max_length=8, choices=Verdict.choices)
    labels = models.JSONField(help_text="{class: 0 or 1}, the reviewer's answer")
    comment = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        db_table = "feedback"
        ordering = ["-created_at"]
        verbose_name_plural = "feedback"

    def __str__(self) -> str:
        return f"{self.job_id}: {self.verdict}"
