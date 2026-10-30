from django.apps import AppConfig


class RegistryConfig(AppConfig):
    """Phase 2: the champion lookup and `seed_model`. Phase 3 adds model_candidates,
    model_approvals, and the approve / reject / rollback admin actions."""

    name = "cxr_backend.registry"
    label = "registry"
