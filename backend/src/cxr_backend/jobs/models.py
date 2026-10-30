"""An upload becomes an `InferenceJob`; the worker's answer becomes its `Prediction`.

A prediction row carries everything needed to trace it back: `model_version` and
`mlflow_run_id` lead to the registry and the training run, `dataset_rev` to the DVC data,
and `preprocess_version` to the `cxr_core.preprocess` code that built the model input.
"""

import uuid

from django.db import models


class InferenceJob(models.Model):
    class Status(models.TextChoices):
        QUEUED = "queued"
        SUCCEEDED = "succeeded"
        FAILED = "failed"

    class Format(models.TextChoices):
        PNG = "png"
        JPEG = "jpeg"
        DICOM = "dicom"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    status = models.CharField(
        max_length=16, choices=Status.choices, default=Status.QUEUED, db_index=True
    )
    filename = models.CharField(max_length=255, help_text="As uploaded; display only")
    format = models.CharField(max_length=8, choices=Format.choices)
    size_bytes = models.PositiveIntegerField()
    sha256 = models.CharField(max_length=64)
    raw_key = models.CharField(max_length=512, help_text="Key in the inference bucket")
    error = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "inference_jobs"
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.id} ({self.status})"


class Prediction(models.Model):
    job = models.OneToOneField(
        InferenceJob, primary_key=True, on_delete=models.CASCADE, related_name="prediction"
    )
    model_name = models.CharField(max_length=128)
    model_version = models.CharField(max_length=32, db_index=True)
    preprocess_version = models.CharField(max_length=32)
    mlflow_run_id = models.CharField(max_length=64, blank=True)
    dataset_rev = models.CharField(max_length=64, blank=True)
    classes = models.JSONField(help_text="Class names in model output order")
    probabilities = models.JSONField(help_text="{class: probability}")
    # The thresholds of the model version that produced this row, copied so the row stays
    # interpretable after the champion changes.
    thresholds = models.JSONField(help_text="{class: threshold}")
    image_key = models.CharField(max_length=512, help_text="Display image in the inference bucket")
    heatmap_keys = models.JSONField(help_text="{class: CAM PNG key in the inference bucket}")
    latency_ms = models.FloatField(help_text="Worker time: download → preprocess → ONNX → CAM")
    worker = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        db_table = "predictions"
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.job_id} → {self.model_name} v{self.model_version}"

    @property
    def positives(self) -> list[str]:
        return [c for c in self.classes if self.probabilities[c] >= self.thresholds[c]]
