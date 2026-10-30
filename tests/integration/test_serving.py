"""The serving side of `core`: gateway routing and where failed jobs end up. The full
upload → prediction path is scripts/smoke_core.sh, which needs a seeded champion."""

import json
import os
import urllib.error
import urllib.request

import pytest

pytestmark = pytest.mark.integration


def _get(url: str, headers: dict[str, str] | None = None) -> tuple[int, str]:
    request = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


def test_gateway_serves_the_ui_and_the_api(gateway_url: str) -> None:
    status, body = _get(gateway_url + "/")
    assert status == 200
    assert '<div id="app">' in body
    assert _get(gateway_url + "/api/health") == (200, '{"status": "ok"}')
    assert _get(gateway_url + "/admin/login/")[0] == 200


@pytest.mark.parametrize("path", ["/internal/champion", "/internal/results", "/metrics"])
def test_gateway_does_not_expose_backend_only_routes(gateway_url: str, path: str) -> None:
    assert _get(gateway_url + path)[0] == 404


def test_rejected_inference_jobs_are_dead_lettered(dotenv: dict[str, str]) -> None:
    """The worker declares `inference`; the policy from definitions.json must apply to it,
    or poison jobs would be dropped instead of kept in `dlq`."""
    import base64

    token = base64.b64encode(f"{dotenv['RABBITMQ_USER']}:{dotenv['RABBITMQ_PASSWORD']}".encode())
    status, body = _get(
        os.environ.get("RABBITMQ_MGMT_URL", "http://localhost:15672") + "/api/queues/%2F/inference",
        {"Authorization": f"Basic {token.decode()}"},
    )
    assert status == 200, "no `inference` queue: is the worker running?"
    queue = json.loads(body)
    assert queue["durable"]
    assert queue["effective_policy_definition"].get("dead-letter-exchange") == "dlx"
