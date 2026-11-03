from django.apps import AppConfig


class MonitoringConfig(AppConfig):
    """/metrics for Prometheus, and the monitor flow's drift reports (ADR 0008)."""

    name = "cxr_backend.monitoring"
    label = "monitoring"
