"""The two Ninja APIs. Public: uploads, results, feedback. Internal: worker callbacks."""

import hmac

from django.conf import settings
from django.db import connection
from django.http import HttpRequest
from ninja import NinjaAPI, Schema
from ninja.security import HttpBearer

from cxr_backend.feedback.api import router as feedback_router
from cxr_backend.jobs.api import internal_router as jobs_internal_router
from cxr_backend.jobs.api import router as jobs_router
from cxr_backend.registry.api import internal_router as registry_internal_router


class InternalToken(HttpBearer):
    def authenticate(self, request: HttpRequest, token: str) -> bool:
        return hmac.compare_digest(token.encode(), settings.INTERNAL_API_TOKEN.encode())


class Health(Schema):
    status: str


public_api = NinjaAPI(title="CXR API", version="1", urls_namespace="api")
public_api.add_router("/jobs", jobs_router)
public_api.add_router("/jobs", feedback_router)


@public_api.get("/health", response=Health, tags=["health"])
def health(request: HttpRequest) -> Health:
    with connection.cursor() as cur:
        cur.execute("SELECT 1")
    return Health(status="ok")


internal_api = NinjaAPI(
    title="CXR internal API",
    version="1",
    urls_namespace="internal",
    auth=InternalToken(),
    docs_url=None,
)
internal_api.add_router("/", jobs_internal_router)
internal_api.add_router("/", registry_internal_router)
