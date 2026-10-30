"""Process-wide clients, built on first use so importing the worker needs no env."""

from dataclasses import dataclass
from functools import cache
from typing import Any

import boto3

from cxr_worker.backend import BackendClient
from cxr_worker.model import ModelHolder
from cxr_worker.settings import Settings


@dataclass(frozen=True)
class Context:
    settings: Settings
    s3: Any
    backend: BackendClient
    holder: ModelHolder


@cache
def context() -> Context:
    settings = Settings.from_env()
    s3 = boto3.client("s3")  # endpoint and credentials from AWS_* (ADR 0001)
    backend = BackendClient(settings.backend_url, settings.internal_token)
    return Context(settings, s3, backend, ModelHolder(settings.model_cache, s3, backend))
