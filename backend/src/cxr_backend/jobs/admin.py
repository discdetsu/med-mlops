from django.contrib import admin
from django.db.models import QuerySet
from django.http import HttpRequest

from cxr_backend.jobs.models import InferenceJob, Prediction


class PredictionInline(admin.StackedInline):  # type: ignore[type-arg]
    model = Prediction
    can_delete = False
    extra = 0
    readonly_fields = [f.name for f in Prediction._meta.fields]


@admin.register(InferenceJob)
class InferenceJobAdmin(admin.ModelAdmin):  # type: ignore[type-arg]
    list_display = ["id", "status", "filename", "format", "model_version", "created_at"]
    list_filter = ["status", "format", "prediction__model_version"]
    search_fields = ["id", "filename", "sha256"]
    readonly_fields = [f.name for f in InferenceJob._meta.fields]
    inlines = [PredictionInline]

    @admin.display(description="model version")
    def model_version(self, job: InferenceJob) -> str:
        prediction = getattr(job, "prediction", None)
        return prediction.model_version if prediction else "-"

    def get_queryset(self, request: HttpRequest) -> QuerySet[InferenceJob]:
        return super().get_queryset(request).select_related("prediction")

    def has_add_permission(self, request: HttpRequest, obj: object = None) -> bool:
        return False


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):  # type: ignore[type-arg]
    list_display = ["job", "model_name", "model_version", "dataset_rev", "latency_ms", "created_at"]
    list_filter = ["model_version", "preprocess_version"]
    search_fields = ["job__id", "mlflow_run_id", "dataset_rev"]
    readonly_fields = [f.name for f in Prediction._meta.fields]

    def has_add_permission(self, request: HttpRequest, obj: object = None) -> bool:
        return False
