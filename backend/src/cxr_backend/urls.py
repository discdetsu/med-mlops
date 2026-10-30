from django.contrib import admin
from django.urls import path

from cxr_backend.api import internal_api, public_api
from cxr_backend.monitoring.views import metrics

admin.site.site_header = "CXR MLOps admin (not for clinical use)"

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", public_api.urls),
    # Not proxied by Nginx: reachable only on the compose network, and token-protected.
    path("internal/", internal_api.urls),
    path("metrics", metrics),
]
