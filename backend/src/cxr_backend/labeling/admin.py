"""The labeling loop in Django admin: read-only rows written by the monitor and export flows."""

from django.contrib import admin
from django.http import HttpRequest

from cxr_backend.labeling.models import LabelingTask, RetrainDecision


class _ReadOnlyAdmin(admin.ModelAdmin):  # type: ignore[type-arg]
    def has_add_permission(self, request: HttpRequest, obj: object = None) -> bool:
        return False

    def has_change_permission(self, request: HttpRequest, obj: object = None) -> bool:
        return False

    def has_delete_permission(self, request: HttpRequest, obj: object = None) -> bool:
        return False


@admin.register(LabelingTask)
class LabelingTaskAdmin(_ReadOnlyAdmin):
    list_display = [
        "created_at",
        "status",
        "reason",
        "model_version",
        "ls_task_id",
        "image_id",
        "dataset_rev",
        "quarantine_reason",
    ]
    list_filter = ["status", "reason", "model_version"]
    search_fields = ["sha256", "image_id", "dataset_rev"]


@admin.register(RetrainDecision)
class RetrainDecisionAdmin(_ReadOnlyAdmin):
    list_display = [
        "created_at",
        "dataset_rev",
        "baseline_rev",
        "new_labels",
        "min_new_labels",
        "triggered",
        "reason",
        "train_flow_run_id",
    ]
    list_filter = ["triggered"]
