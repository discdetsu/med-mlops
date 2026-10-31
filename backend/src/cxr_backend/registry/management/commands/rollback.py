from argparse import ArgumentParser
from typing import Any

from django.core.management.base import BaseCommand, CommandError
from mlflow.exceptions import MlflowException

from cxr_backend.registry import promotion


class Command(BaseCommand):
    help = "Move @champion back to @previous, the same as the admin's rollback button."

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--reason", required=True, help="recorded in model_approvals")
        parser.add_argument("--by", default="cli", help="actor recorded in model_approvals")

    def handle(self, *args: Any, **options: Any) -> None:
        try:
            decision = promotion.rollback(reason=options["reason"], actor=options["by"])
        except (promotion.PromotionError, MlflowException) as exc:
            raise CommandError(str(exc)) from exc
        approval = decision.approval
        self.stdout.write(
            f"rolled back: version {approval.to_version} is @champion (was "
            f"{approval.from_version})"
            + ("" if decision.event_published else "; workers NOT notified, restart them")
        )
