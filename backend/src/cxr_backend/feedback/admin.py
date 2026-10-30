from django.contrib import admin
from django.http import HttpRequest

from cxr_backend.feedback.models import Feedback


@admin.register(Feedback)
class FeedbackAdmin(admin.ModelAdmin):  # type: ignore[type-arg]
    list_display = ["job", "verdict", "model_version", "created_at"]
    list_filter = ["verdict", "model_version"]
    readonly_fields = [f.name for f in Feedback._meta.fields]

    def has_add_permission(self, request: HttpRequest, obj: object = None) -> bool:
        return False
