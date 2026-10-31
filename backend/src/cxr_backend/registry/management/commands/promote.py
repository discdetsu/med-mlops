from argparse import ArgumentParser
from typing import Any

from django.core.management.base import BaseCommand, CommandError
from mlflow.exceptions import MlflowException

from cxr_backend.registry import promotion
from cxr_backend.registry.models import ModelCandidate
from cxr_core.registry import MODEL_NAME


class Command(BaseCommand):
    help = (
        "Approve a gated candidate, the same as the admin's button (`promote VERSION --reason "
        "...`), or the newest candidate the gate marked for auto_promote (`promote --auto`)."
    )

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("version", nargs="?", help="model version to approve")
        parser.add_argument("--reason", default="", help="recorded in model_approvals")
        parser.add_argument("--by", default="cli", help="actor recorded in model_approvals")
        parser.add_argument(
            "--auto",
            action="store_true",
            help="approve the newest pending candidate with auto_promote set; a no-op if none",
        )

    def handle(self, *args: Any, **options: Any) -> None:
        version, auto = options["version"], options["auto"]
        if bool(version) == auto:
            raise CommandError("give a VERSION or --auto, not both")
        try:
            if auto:
                decision = promotion.auto_promote()
                if decision is None:
                    self.stdout.write("auto_promote: no passed candidate marked for it")
                    return
            else:
                candidate = ModelCandidate.objects.filter(
                    model_name=MODEL_NAME, model_version=version
                ).first()
                if candidate is None:
                    raise CommandError(f"no candidate for version {version}; run `make gate`")
                decision = promotion.approve(
                    candidate.pk, reason=options["reason"], actor=options["by"]
                )
        except (promotion.PromotionError, MlflowException) as exc:
            raise CommandError(str(exc)) from exc
        approval = decision.approval
        self.stdout.write(
            f"version {approval.to_version} is @champion (was {approval.from_version})"
            + ("" if decision.event_published else "; workers NOT notified, restart them")
        )
