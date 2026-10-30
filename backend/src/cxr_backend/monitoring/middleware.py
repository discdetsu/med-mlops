import time
from collections.abc import Callable

from django.http import HttpRequest, HttpResponse

from cxr_backend.monitoring.metrics import HTTP_REQUEST_DURATION


class RequestMetricsMiddleware:
    """Times every request, labelled by the URL pattern (not the path) to bound cardinality."""

    def __init__(self, get_response: Callable[[HttpRequest], HttpResponse]) -> None:
        self.get_response = get_response

    def __call__(self, request: HttpRequest) -> HttpResponse:
        start = time.perf_counter()
        response = self.get_response(request)
        match = request.resolver_match
        route = match.route if match is not None else "unmatched"
        if route != "metrics":
            HTTP_REQUEST_DURATION.labels(request.method, route, response.status_code).observe(
                time.perf_counter() - start
            )
        return response
