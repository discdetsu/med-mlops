"""The small part of Label Studio's API the labeling loop uses.

It logs in with the admin account in .env. Label Studio 1.23 disables legacy API tokens for
organizations unless they are created with a CLI flag, and its personal tokens can only be created
after logging in, so a session works on any existing database with no extra secret.

Images stay in the `inference` bucket, not in the DVC remote, whose file names are hashes. Label
Studio proxies them with `presign: false`: a presigned URL would name `http://rustfs:9000`, which a
browser can't reach. The S3 endpoint comes from `S3_ENDPOINT` on the label-studio service, which its
SSRF guard trusts; an endpoint set through this API would be refused as a private address.
"""

import os
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Self
from xml.sax.saxutils import quoteattr

import httpx

FROM_NAME = "findings"
TO_NAME = "image"
# Required, so an empty answer always means the reviewer never answered, not "nothing found".
NONE_CHOICE = "None of these"
_PAGE_SIZE = 100


class LabelStudioError(RuntimeError):
    """Label Studio refused a request or the login."""


def s3_settings() -> dict[str, str]:
    """The credentials the project's storage connection needs; the endpoint is its own setting."""
    return {
        "aws_access_key_id": os.environ["AWS_ACCESS_KEY_ID"],
        "aws_secret_access_key": os.environ["AWS_SECRET_ACCESS_KEY"],
        "region_name": os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    }


def label_config(classes: Sequence[str]) -> str:
    """Multi-label choices over the canonical classes. No model output is shown: a reviewer who
    sees the prediction anchors on it (ADR 0008)."""
    choices = "".join(f"<Choice value={quoteattr(c)}/>" for c in [*classes, NONE_CHOICE])
    return (
        f'<View><Image name="{TO_NAME}" value="$image" zoom="true"/>'
        f'<Choices name="{FROM_NAME}" toName="{TO_NAME}" choice="multiple" required="true">'
        f"{choices}</Choices></View>"
    )


def choices_result(choices: Iterable[str]) -> list[dict[str, Any]]:
    """An annotation's `result` for these choices."""
    return [
        {
            "from_name": FROM_NAME,
            "to_name": TO_NAME,
            "type": "choices",
            "value": {"choices": list(choices)},
        }
    ]


class LabelStudio:
    def __init__(self, client: httpx.Client, username: str, password: str) -> None:
        self._client = client
        self._login(username, password)

    @classmethod
    def from_env(cls) -> Self | None:
        """None without LABEL_STUDIO_URL: Label Studio is only in the `full` profile."""
        url = os.environ.get("LABEL_STUDIO_URL")
        if not url:
            return None
        try:
            username = os.environ["LABEL_STUDIO_USERNAME"]
            password = os.environ["LABEL_STUDIO_PASSWORD"]
        except KeyError as exc:
            raise LabelStudioError(f"{exc.args[0]} is not set (see .env.example)") from None
        return cls(httpx.Client(base_url=url, timeout=60), username, password)

    def _login(self, username: str, password: str) -> None:
        self._client.get("/user/login/").raise_for_status()
        response = self._client.post(
            "/user/login/",
            data={
                "email": username,
                "password": password,
                "csrfmiddlewaretoken": self._client.cookies.get("csrftoken") or "",
            },
            headers={"Referer": f"{self._client.base_url}user/login/"},
        )
        if response.status_code != 302:  # a failed login shows the form again with a 200
            raise LabelStudioError(f"login as {username} failed (HTTP {response.status_code})")

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        headers = {
            "X-CSRFToken": self._client.cookies.get("csrftoken") or "",
            "Referer": str(self._client.base_url),
        }
        response = self._client.request(method, path, headers=headers, **kwargs)
        if response.is_error:
            raise LabelStudioError(
                f"{method} {path}: HTTP {response.status_code}: {response.text[:300]}"
            )
        return response.json() if response.content else None

    def ensure_project(
        self, title: str, classes: Sequence[str], s3: Mapping[str, str], bucket: str
    ) -> int:
        """The project with this title and its storage connection, created when missing."""
        found = self._request("GET", "/api/projects", params={"title": title})["results"]
        existing = [p for p in found if p["title"] == title]
        if existing:
            project = int(existing[0]["id"])
        else:
            created = self._request(
                "POST",
                "/api/projects",
                json={"title": title, "label_config": label_config(classes)},
            )
            project = int(created["id"])
        if not self._request("GET", "/api/storages/s3", params={"project": project}):
            self._request(
                "POST",
                "/api/storages/s3",
                json={
                    "project": project,
                    "title": bucket,
                    "bucket": bucket,
                    "presign": False,
                    "use_blob_urls": False,
                    **s3,
                },
            )
        return project

    def list_tasks(self, project: int) -> list[dict[str, Any]]:
        """Every task of the project with its annotations."""
        tasks: list[dict[str, Any]] = []
        page = 1
        while True:
            body = self._request(
                "GET",
                "/api/tasks",
                params={"project": project, "fields": "all", "page": page, "page_size": _PAGE_SIZE},
            )
            tasks.extend(body["tasks"])
            if len(tasks) >= body["total"] or not body["tasks"]:
                return tasks
            page += 1

    def import_tasks(self, project: int, tasks: Sequence[Mapping[str, Any]]) -> list[int]:
        """Create tasks and return their IDs, in the same order."""
        body = self._request(
            "POST",
            f"/api/projects/{project}/import",
            params={"return_task_ids": "true"},
            json=list(tasks),
        )
        return [int(t) for t in body["task_ids"]]

    def annotate(self, task_id: int, choices: Iterable[str]) -> None:
        self._request(
            "POST",
            f"/api/tasks/{task_id}/annotations",
            json={"result": choices_result(choices), "was_cancelled": False},
        )
