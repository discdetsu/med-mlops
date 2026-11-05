"""The Label Studio client against a fake of its API."""

import json
from collections.abc import Callable
from typing import Any
from xml.etree import ElementTree

import httpx
import pytest

from cxr_flows.labelstudio import (
    FROM_NAME,
    NONE_CHOICE,
    TO_NAME,
    LabelStudio,
    LabelStudioError,
    label_config,
)

BASE = "http://label-studio:8080"
S3 = {"aws_access_key_id": "k", "aws_secret_access_key": "s", "region_name": "r"}


class FakeApi:
    def __init__(self, password: str = "right") -> None:
        self.password = password
        self.projects: list[dict[str, Any]] = []
        self.storages: list[dict[str, Any]] = []
        self.tasks: list[dict[str, Any]] = []
        self.annotations: list[tuple[int, dict[str, Any]]] = []
        self.csrf: list[str | None] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path, method = request.url.path, request.method
        if method != "GET":
            self.csrf.append(request.headers.get("X-CSRFToken"))
        if path == "/user/login/":
            if method == "GET":
                return httpx.Response(200, headers={"Set-Cookie": "csrftoken=first; Path=/"})
            form = dict(p.split("=", 1) for p in request.content.decode().split("&"))
            if form.get("password") != self.password or form.get("csrfmiddlewaretoken") != "first":
                return httpx.Response(200, text="<form>")
            return httpx.Response(
                302, headers={"Location": "/", "Set-Cookie": "csrftoken=rotated; Path=/"}
            )
        if path == "/api/projects":
            if method == "GET":
                title = request.url.params["title"]
                # The real filter matches loosely, so the client has to compare titles itself.
                results = [p for p in self.projects if title in p["title"]]
                return httpx.Response(200, json={"results": results})
            project = {**json.loads(request.content), "id": len(self.projects) + 1}
            self.projects.append(project)
            return httpx.Response(201, json=project)
        if path == "/api/storages/s3":
            project = int(request.url.params["project"]) if method == "GET" else None
            if method == "GET":
                return httpx.Response(
                    200, json=[s for s in self.storages if s["project"] == project]
                )
            self.storages.append(json.loads(request.content))
            return httpx.Response(201, json={"id": len(self.storages)})
        if path == "/api/tasks":
            page, size = int(request.url.params["page"]), int(request.url.params["page_size"])
            mine = self.tasks[(page - 1) * size : page * size]
            if not mine and page > 1:
                return httpx.Response(404, json={"detail": "Invalid page."})
            return httpx.Response(200, json={"tasks": mine, "total": len(self.tasks)})
        if path.endswith("/import"):
            created = [
                {"id": len(self.tasks) + i + 1, **t}
                for i, t in enumerate(json.loads(request.content))
            ]
            self.tasks += created
            return httpx.Response(201, json={"task_ids": [t["id"] for t in created]})
        if path.endswith("/annotations"):
            self.annotations.append((int(path.split("/")[3]), json.loads(request.content)))
            return httpx.Response(201, json={"id": 1})
        return httpx.Response(500, text=f"unexpected {method} {path}")


Connect = Callable[..., LabelStudio]


@pytest.fixture
def api() -> FakeApi:
    return FakeApi()


@pytest.fixture
def connect(api: FakeApi) -> Connect:
    def make(password: str = "right") -> LabelStudio:
        client = httpx.Client(base_url=BASE, transport=httpx.MockTransport(api))
        return LabelStudio(client, "admin@example.com", password)

    return make


def test_login_failure_is_an_error(connect: Connect) -> None:
    with pytest.raises(LabelStudioError, match="login"):
        connect(password="wrong")


def test_requests_send_the_csrf_token_rotated_by_the_login(api: FakeApi, connect: Connect) -> None:
    connect().ensure_project("CXR", ["Mass"], S3, "inference")
    assert api.csrf[1:] == ["rotated", "rotated"]  # after the login POST


def test_project_and_storage_are_created_once(api: FakeApi, connect: Connect) -> None:
    ls = connect()
    first = ls.ensure_project("CXR review", ["Mass"], S3, "inference")
    assert ls.ensure_project("CXR review", ["Mass"], S3, "inference") == first
    assert len(api.projects) == 1 and len(api.storages) == 1
    storage = api.storages[0]
    assert storage["bucket"] == "inference" and storage["presign"] is False
    assert "s3_endpoint" not in storage  # refused by the SSRF guard; S3_ENDPOINT sets it


def test_a_title_containing_another_is_a_different_project(api: FakeApi, connect: Connect) -> None:
    ls = connect()
    other = ls.ensure_project("CXR review old", ["Mass"], S3, "inference")
    assert ls.ensure_project("CXR review", ["Mass"], S3, "inference") != other


def test_storage_is_added_to_a_project_that_lost_it(api: FakeApi, connect: Connect) -> None:
    api.projects.append({"id": 7, "title": "CXR review"})
    assert connect().ensure_project("CXR review", ["Mass"], S3, "inference") == 7
    assert api.storages[0]["project"] == 7


def test_list_tasks_reads_every_page(api: FakeApi, connect: Connect) -> None:
    api.tasks = [{"id": i, "data": {}} for i in range(1, 251)]
    assert [t["id"] for t in connect().list_tasks(1)] == list(range(1, 251))


def test_list_tasks_of_an_empty_project(connect: Connect) -> None:
    assert connect().list_tasks(1) == []


def test_import_returns_ids_in_order(connect: Connect) -> None:
    ls = connect()
    assert ls.import_tasks(1, [{"data": {"a": 1}}, {"data": {"a": 2}}]) == [1, 2]
    assert ls.import_tasks(1, [{"data": {"a": 3}}]) == [3]


def test_annotate_posts_the_choices(api: FakeApi, connect: Connect) -> None:
    connect().annotate(5, ["Mass"])
    task, body = api.annotations[0]
    assert task == 5 and body["was_cancelled"] is False
    assert body["result"][0]["value"] == {"choices": ["Mass"]}
    assert (body["result"][0]["from_name"], body["result"][0]["to_name"]) == (FROM_NAME, TO_NAME)


def test_an_api_error_says_what_failed(api: FakeApi, connect: Connect) -> None:
    with pytest.raises(LabelStudioError, match="unexpected POST"):
        connect()._request("POST", "/api/nope")


def test_from_env_without_a_url_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LABEL_STUDIO_URL", raising=False)
    assert LabelStudio.from_env() is None


def test_from_env_names_a_missing_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LABEL_STUDIO_URL", BASE)
    monkeypatch.delenv("LABEL_STUDIO_USERNAME", raising=False)
    with pytest.raises(LabelStudioError, match="LABEL_STUDIO_USERNAME"):
        LabelStudio.from_env()


def test_label_config_is_multi_choice_required_and_shows_no_prediction() -> None:
    root = ElementTree.fromstring(label_config(["Mass", "A & B"]))
    choices = root.find("Choices")
    assert choices is not None
    assert choices.get("choice") == "multiple" and choices.get("required") == "true"
    assert [c.get("value") for c in choices] == ["Mass", "A & B", NONE_CHOICE]
    assert {child.tag for child in root} == {"Image", "Choices"}
