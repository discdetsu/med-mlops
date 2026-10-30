from django.http import HttpRequest, HttpResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest


def metrics(request: HttpRequest) -> HttpResponse:
    """Prometheus scrape target. Not routed by Nginx; only reachable on the compose network."""
    return HttpResponse(generate_latest(), content_type=CONTENT_TYPE_LATEST)
