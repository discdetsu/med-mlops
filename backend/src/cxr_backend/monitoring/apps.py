from django.apps import AppConfig


class MonitoringConfig(AppConfig):
    """Phase 2: /metrics. Phase 4 adds drift_reports and the delayed-AUROC views."""

    name = "cxr_backend.monitoring"
    label = "monitoring"
