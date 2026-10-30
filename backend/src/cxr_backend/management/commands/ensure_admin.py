import os
from typing import Any

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Create the admin superuser from DJANGO_ADMIN_USER / DJANGO_ADMIN_PASSWORD once."

    def handle(self, *args: Any, **options: Any) -> None:
        username = os.environ.get("DJANGO_ADMIN_USER")
        password = os.environ.get("DJANGO_ADMIN_PASSWORD")
        if not username or not password:
            self.stdout.write("DJANGO_ADMIN_USER/PASSWORD not set; no admin user created")
            return
        users = get_user_model().objects
        if users.filter(username=username).exists():
            # Never reset a password someone may have changed in the admin.
            return
        users.create_superuser(username=username, email="", password=password)
        self.stdout.write(f"created admin user {username!r}")
