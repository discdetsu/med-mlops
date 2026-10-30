"""Settings for the unit and API contract tests: SQLite in memory, no services needed.
S3, Celery, and MLflow are replaced per test (see backend/tests/conftest.py)."""

import os

os.environ.setdefault("DJANGO_SECRET_KEY", "test-only")
os.environ.setdefault("APP_DATABASE_URL", "sqlite://")
os.environ.setdefault("INTERNAL_API_TOKEN", "test-internal-token")
os.environ.setdefault("CELERY_BROKER_URL", "memory://")

from cxr_backend.settings import *  # noqa: F403

# Always SQLite, even when the Makefile has exported the real APP_DATABASE_URL.
DATABASES = {"default": {"ENGINE": "django.db.backends.sqlite3", "NAME": ":memory:"}}
# No collected static files in tests; WhiteNoise would warn about the missing STATIC_ROOT.
MIDDLEWARE = [m for m in MIDDLEWARE if not m.startswith("whitenoise.")]  # noqa: F405
STORAGES = {
    "default": {"BACKEND": "django.core.files.storage.InMemoryStorage"},
    "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
}
