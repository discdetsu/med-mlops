"""Django settings. Everything that differs between machines comes from env vars, which
docker-compose.yml fills from .env; see .env.example for the full list."""

import os
from pathlib import Path
from urllib.parse import unquote, urlsplit

from django.core.exceptions import ImproperlyConfigured


def _require(name: str) -> str:
    try:
        return os.environ[name]
    except KeyError:
        raise ImproperlyConfigured(f"{name} is not set (see .env.example)") from None


def _database(url: str) -> dict[str, object]:
    parts = urlsplit(url)
    if parts.scheme == "sqlite":
        return {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": parts.path.lstrip("/") or ":memory:",
        }
    if parts.scheme not in ("postgres", "postgresql"):
        raise ImproperlyConfigured(f"unsupported database URL scheme {parts.scheme!r}")
    return {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": parts.path.lstrip("/"),
        "USER": unquote(parts.username or ""),
        "PASSWORD": unquote(parts.password or ""),
        "HOST": parts.hostname or "",
        "PORT": str(parts.port or 5432),
        "CONN_MAX_AGE": 60,
        "CONN_HEALTH_CHECKS": True,
    }


BASE_DIR = Path(__file__).resolve().parent

SECRET_KEY = _require("DJANGO_SECRET_KEY")
DEBUG = os.environ.get("DJANGO_DEBUG") == "1"
ALLOWED_HOSTS = os.environ.get("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1,backend").split(",")
# Nginx forwards the browser's Host (with port), so the admin's CSRF origin check passes
# without this for http://localhost:8000; it is here for other hostnames.
CSRF_TRUSTED_ORIGINS = [
    o for o in os.environ.get("DJANGO_CSRF_TRUSTED_ORIGINS", "").split(",") if o
]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "cxr_backend",  # management commands only
    "cxr_backend.jobs",
    "cxr_backend.feedback",
    "cxr_backend.registry",
    "cxr_backend.monitoring",
]

MIDDLEWARE = [
    "cxr_backend.monitoring.middleware.RequestMetricsMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "cxr_backend.urls"
WSGI_APPLICATION = "cxr_backend.wsgi.application"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

DATABASES = {"default": _database(_require("APP_DATABASE_URL"))}
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_TZ = True

# The admin's static files, served by WhiteNoise; Nginx only proxies /static/ through.
STATIC_URL = "/static/"
STATIC_ROOT = Path(os.environ.get("DJANGO_STATIC_ROOT", "/app/static"))

# Uploads above this are rejected by the API; Nginx's client_max_body_size matches it.
MAX_UPLOAD_BYTES = 32 * 1024 * 1024
DATA_UPLOAD_MAX_MEMORY_SIZE = MAX_UPLOAD_BYTES + 1024 * 1024
FILE_UPLOAD_MAX_MEMORY_SIZE = 5 * 1024 * 1024

# Shared secret for /internal/* (worker → backend). /internal is also never routed by Nginx.
INTERNAL_API_TOKEN = _require("INTERNAL_API_TOKEN")

CELERY_BROKER_URL = _require("CELERY_BROKER_URL")
INFERENCE_BUCKET = "inference"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {"console": {"class": "logging.StreamHandler"}},
    "root": {"handlers": ["console"], "level": "INFO"},
}
