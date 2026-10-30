from argparse import ArgumentParser
from typing import Any

from django.core.management.base import BaseCommand, CommandError

from cxr_backend.registry.seed import SeedError, seed


class Command(BaseCommand):
    help = "Register a seed bundle (URL or path) as @champion and tell the workers to load it."

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "source", help="https:// URL or local path of cxr-seed-bundle-v*.tar.gz"
        )
        parser.add_argument("--sha256", help="expected digest; default: the .sha256 next to it")
        parser.add_argument(
            "--force", action="store_true", help="register it even if a @champion exists"
        )

    def handle(self, *args: Any, **options: Any) -> None:
        try:
            result = seed(options["source"], options["sha256"], options["force"])
        except (SeedError, OSError) as exc:
            raise CommandError(str(exc)) from exc
        if result.created:
            self.stdout.write(f"registered version {result.model_version} as @champion")
        else:
            self.stdout.write(
                f"@champion is already version {result.model_version}; "
                "pass --force to register the bundle anyway"
            )
