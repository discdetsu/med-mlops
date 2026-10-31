from django.apps import AppConfig


class RegistryConfig(AppConfig):
    """The champion lookup, `seed_model`, and promotion: model candidates from the gate,
    approve / reject / rollback in admin, and the model_approvals log (ADR 0007)."""

    name = "cxr_backend.registry"
    label = "registry"
