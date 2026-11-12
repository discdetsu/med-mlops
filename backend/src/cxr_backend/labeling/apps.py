from django.apps import AppConfig


class LabelingConfig(AppConfig):
    """The labeling loop's records: tasks sent to Label Studio and retrain decisions (ADR 0009)."""

    name = "cxr_backend.labeling"
    label = "labeling"
