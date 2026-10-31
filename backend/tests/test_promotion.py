"""Promotion (ADR 0007): the service in registry.promotion, its admin pages, and the
management commands, against SQLite and an in-memory MLflow registry."""

from typing import Any

import pytest
from django.contrib.auth.models import Permission, User
from django.core.management import CommandError, call_command
from django.db import DatabaseError, connection
from django.test import Client
from mlflow.exceptions import MlflowException

import cxr_backend.registry.promotion as promotion
from cxr_backend.registry.models import ModelApproval, ModelCandidate
from cxr_core.messaging import ChampionChanged

from .conftest import FakeMlflow, FakeS3

pytestmark = pytest.mark.django_db

REPORT: dict[str, Any] = {
    "passed": True,
    "champion_version": "1",
    "challenger_version": "2",
    "n": 1000,
    "n_patients": 400,
    "checks": [
        {"name": "macro_auroc", "passed": True, "detail": "lower 95% bound is -0.0040"},
        {"name": "subgroups", "passed": True, "detail": "<script>alert(1)</script>"},
    ],
    "macro_auroc": {
        "champion": 0.70,
        "challenger": 0.72,
        "delta": 0.02,
        "delta_ci": [-0.004, 0.041],
    },
    "classes": [
        {"class": "A", "champion": 0.7, "challenger": 0.72, "delta": 0.02, "delta_ci": None}
    ],
    "subgroups": [
        {
            "attribute": "view",
            "value": "AP",
            "n": 300,
            "champion": 0.6,
            "challenger": 0.61,
            "delta": 0.01,
            "delta_ci": [-0.02, 0.04],
            "judged": True,
            "passed": True,
        }
    ],
    "ece": {"champion": 0.01, "challenger": 0.012, "delta": 0.002},
    "config": {"confidence": 0.95},
}
REPORT_URI = "s3://artifacts/models/cxr-classifier/2/gate/vs-champion-v1.html"


@pytest.fixture
def mlflow(monkeypatch: pytest.MonkeyPatch) -> FakeMlflow:
    """Version 1 is @champion, version 2 @challenger."""
    fake = FakeMlflow(champion="1")
    fake.versions["2"] = {"bundle_uri": "s3://artifacts/models/cxr-classifier/2/"}
    fake.aliases["challenger"] = "2"
    monkeypatch.setattr(promotion, "mlflow_client", lambda: fake)
    return fake


@pytest.fixture
def events(monkeypatch: pytest.MonkeyPatch) -> list[ChampionChanged]:
    sent: list[ChampionChanged] = []
    monkeypatch.setattr(promotion, "publish_champion_changed", sent.append)
    return sent


def candidate(version: str = "2", **overrides: Any) -> ModelCandidate:
    fields: dict[str, Any] = {
        "model_name": "cxr-classifier",
        "model_version": version,
        "champion_version": "1",
        "status": ModelCandidate.Status.PENDING,
        "gate_passed": True,
        "report": REPORT,
        "report_uri": REPORT_URI,
        "mlflow_run_id": "run-2",
        "dataset_rev": "f774f7a855ab",
    }
    return ModelCandidate.objects.create(**(fields | overrides))


def bundle_uri(version: str) -> str:
    return f"s3://artifacts/models/cxr-classifier/{version}/"


# ---------------------------------------------------------------------- service


def test_approve_promotes_the_candidate(mlflow: FakeMlflow, events: list[ChampionChanged]) -> None:
    c = candidate()
    admin = User.objects.create_user("alice")

    decision = promotion.approve(c.pk, reason="better on AP", user=admin)

    assert mlflow.aliases == {"champion": "2", "previous": "1"}
    assert mlflow.versions["2"]["promotion_status"] == "approved"
    c.refresh_from_db()
    assert c.status == ModelCandidate.Status.APPROVED and c.decided_at is not None
    approval = decision.approval
    assert (approval.action, approval.from_version, approval.to_version) == ("approve", "1", "2")
    assert (approval.user, approval.actor, approval.reason) == (admin, "alice", "better on AP")
    assert approval.event_published
    assert events == [ChampionChanged("cxr-classifier", "2", bundle_uri("2"))]


def test_approve_refuses_a_candidate_gated_against_another_champion(
    mlflow: FakeMlflow, events: list[ChampionChanged]
) -> None:
    c = candidate(champion_version="0")
    with pytest.raises(promotion.PromotionError, match="re-run the gate"):
        promotion.approve(c.pk, reason="ok")
    assert mlflow.aliases["champion"] == "1"
    assert not ModelApproval.objects.exists()
    assert not events


@pytest.mark.parametrize(
    "overrides",
    [
        {"status": ModelCandidate.Status.GATE_FAILED, "gate_passed": False},
        {"status": ModelCandidate.Status.REJECTED},
        {"status": ModelCandidate.Status.APPROVED},
    ],
)
def test_approve_refuses_candidates_that_are_not_pending(
    mlflow: FakeMlflow, events: list[ChampionChanged], overrides: dict[str, Any]
) -> None:
    c = candidate(**overrides)
    with pytest.raises(promotion.PromotionError):
        promotion.approve(c.pk, reason="ok")
    assert mlflow.aliases["champion"] == "1"


def test_every_decision_needs_a_reason(mlflow: FakeMlflow, events: list[ChampionChanged]) -> None:
    c = candidate()
    with pytest.raises(promotion.PromotionError, match="reason"):
        promotion.approve(c.pk, reason="   ")
    with pytest.raises(promotion.PromotionError, match="reason"):
        promotion.rollback(reason="")
    assert mlflow.aliases["champion"] == "1"


def test_a_failed_event_still_records_the_approval(
    mlflow: FakeMlflow, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(event: ChampionChanged) -> None:
        raise OSError("rabbitmq down")

    monkeypatch.setattr(promotion, "publish_champion_changed", broken)
    decision = promotion.approve(candidate().pk, reason="ok")

    assert mlflow.aliases["champion"] == "2"
    assert not decision.event_published
    assert not ModelApproval.objects.get().event_published


def _nothing_was_decided(c: ModelCandidate) -> None:
    c.refresh_from_db()
    assert c.status == ModelCandidate.Status.PENDING and c.decided_at is None
    assert not ModelApproval.objects.exists()


@pytest.mark.parametrize("alias", ["champion", "previous"])
def test_a_failed_alias_move_leaves_the_registry_and_records_unchanged(
    mlflow: FakeMlflow, events: list[ChampionChanged], alias: str
) -> None:
    mlflow.versions["0"] = {}
    mlflow.aliases["previous"] = "0"  # an earlier rollback target that must survive
    before = dict(mlflow.aliases)
    c = candidate()
    mlflow.fail.add(("set", alias))

    with pytest.raises(MlflowException, match="injected"):
        promotion.approve(c.pk, reason="ok")

    assert mlflow.aliases == before
    _nothing_was_decided(c)
    assert not events


@pytest.mark.django_db(transaction=True)
def test_a_failed_commit_puts_the_aliases_back(
    mlflow: FakeMlflow, events: list[ChampionChanged], monkeypatch: pytest.MonkeyPatch
) -> None:
    before = dict(mlflow.aliases)
    c = candidate()

    def disk_full() -> None:
        raise DatabaseError("disk full")

    with monkeypatch.context() as m:
        m.setattr(connection, "commit", disk_full)
        with pytest.raises(DatabaseError, match="disk full"):
            promotion.approve(c.pk, reason="ok")

    assert mlflow.aliases == before
    _nothing_was_decided(c)
    assert not events


def test_reject_closes_the_candidate_and_drops_challenger(
    mlflow: FakeMlflow, events: list[ChampionChanged]
) -> None:
    c = candidate()
    decision = promotion.reject(c.pk, reason="worse on AP")

    c.refresh_from_db()
    assert c.status == ModelCandidate.Status.REJECTED
    assert mlflow.aliases == {"champion": "1"}
    assert decision.approval.action == "reject"
    assert not events
    with pytest.raises(promotion.PromotionError, match="only pending"):
        promotion.reject(c.pk, reason="again")


def test_rollback_restores_the_previous_champion_once(
    mlflow: FakeMlflow, events: list[ChampionChanged]
) -> None:
    c = candidate()
    promotion.approve(c.pk, reason="ok")

    decision = promotion.rollback(reason="false positives on AP", actor="bob")

    assert mlflow.aliases == {"champion": "1"}
    c.refresh_from_db()
    assert c.status == ModelCandidate.Status.ROLLED_BACK
    approval = decision.approval
    assert (approval.action, approval.from_version, approval.to_version) == ("rollback", "2", "1")
    assert (approval.candidate, approval.actor) == (c, "bob")
    assert events[-1] == ChampionChanged("cxr-classifier", "1", bundle_uri("1"))
    with pytest.raises(promotion.PromotionError, match="nothing to roll back"):
        promotion.rollback(reason="again")


def test_a_failed_previous_clear_undoes_the_rollback(
    mlflow: FakeMlflow, events: list[ChampionChanged]
) -> None:
    c = candidate()
    promotion.approve(c.pk, reason="ok")
    mlflow.fail.add(("delete", "previous"))

    with pytest.raises(MlflowException, match="injected"):
        promotion.rollback(reason="bad")

    assert mlflow.aliases == {"champion": "2", "previous": "1"}
    c.refresh_from_db()
    assert c.status == ModelCandidate.Status.APPROVED
    assert ModelApproval.objects.count() == 1


def test_rollback_refuses_when_previous_is_already_champion(
    mlflow: FakeMlflow, events: list[ChampionChanged]
) -> None:
    """E.g. someone moved the aliases by hand in MLflow: a rollback would change nothing."""
    c = candidate()
    promotion.approve(c.pk, reason="ok")
    mlflow.aliases["previous"] = "2"

    with pytest.raises(promotion.PromotionError, match="already @champion"):
        promotion.rollback(reason="again")

    c.refresh_from_db()
    assert c.status == ModelCandidate.Status.APPROVED
    assert ModelApproval.objects.count() == 1


def test_auto_promote_takes_only_candidates_marked_for_it(
    mlflow: FakeMlflow, events: list[ChampionChanged]
) -> None:
    candidate()
    assert promotion.auto_promote() is None
    assert mlflow.aliases["champion"] == "1"

    mlflow.versions["3"] = {}
    mlflow.aliases["challenger"] = "3"
    candidate("3", auto_promote=True)
    decision = promotion.auto_promote()

    assert decision is not None
    assert decision.approval.actor == "auto_promote"
    assert mlflow.aliases["champion"] == "3"


def test_auto_promote_ignores_an_older_marked_candidate(
    mlflow: FakeMlflow, events: list[ChampionChanged]
) -> None:
    """The run that just finished registered @challenger 2 and it failed the gate; an older
    pending candidate marked for auto_promote must not be promoted in its place."""
    mlflow.versions["3"] = {}
    candidate("3", auto_promote=True)
    candidate("2", status=ModelCandidate.Status.GATE_FAILED, gate_passed=False)

    assert promotion.auto_promote() is None
    assert mlflow.aliases["champion"] == "1"

    decision = promotion.auto_promote(version="3")
    assert decision is not None and mlflow.aliases["champion"] == "3"


def test_regating_reopens_a_rejected_candidate(
    mlflow: FakeMlflow, events: list[ChampionChanged]
) -> None:
    """The gate flow's upsert resets status and decided_at; approval works again."""
    c = candidate()
    promotion.reject(c.pk, reason="flaky")
    ModelCandidate.objects.filter(pk=c.pk).update(status="pending", decided_at=None)
    promotion.approve(c.pk, reason="re-gated")
    assert mlflow.aliases["champion"] == "2"
    assert list(c.decisions.values_list("action", flat=True).order_by("created_at")) == [
        "reject",
        "approve",
    ]


# ---------------------------------------------------------------------- admin


def test_candidate_page_shows_the_report_and_the_buttons(
    admin_client: Client, mlflow: FakeMlflow
) -> None:
    c = candidate()
    page = admin_client.get(f"/admin/registry/modelcandidate/{c.pk}/change/").content.decode()

    assert "Gate PASSED" in page
    assert "+0.0200 (-0.0040 to +0.0410)" in page
    assert "&lt;script&gt;" in page and "<script>alert" not in page
    assert f"/admin/registry/modelcandidate/{c.pk}/approve/" in page
    assert f"/admin/registry/modelcandidate/{c.pk}/reject/" in page
    assert "@champion: version 1" in page


def test_a_stale_candidate_shows_why_it_cant_be_approved(
    admin_client: Client, mlflow: FakeMlflow
) -> None:
    c = candidate(champion_version="0")
    page = admin_client.get(f"/admin/registry/modelcandidate/{c.pk}/change/").content.decode()
    assert "Can't approve" in page and "re-run the gate" in page
    assert f"/{c.pk}/approve/" not in page


def test_changelist_shows_aliases_and_rollback(
    admin_client: Client, mlflow: FakeMlflow, events: list[ChampionChanged]
) -> None:
    candidate()
    page = admin_client.get("/admin/registry/modelcandidate/").content.decode()
    assert "@champion: version 1" in page
    assert "+0.0200" in page
    assert "Roll back" not in page  # no @previous yet

    mlflow.versions["0"] = {}
    mlflow.aliases["previous"] = "0"
    page = admin_client.get("/admin/registry/modelcandidate/").content.decode()
    assert "Roll back @champion to version 0" in page

    mlflow.aliases["previous"] = "1"  # same as @champion: rolling back would change nothing
    page = admin_client.get("/admin/registry/modelcandidate/").content.decode()
    assert "Roll back" not in page


def test_approving_through_the_admin(
    admin_client: Client, mlflow: FakeMlflow, events: list[ChampionChanged]
) -> None:
    c = candidate()
    url = f"/admin/registry/modelcandidate/{c.pk}/approve/"
    form = admin_client.get(url).content.decode()
    assert "@champion moves from version 1 to version 2" in form

    assert admin_client.post(url, {"reason": ""}).status_code == 200  # form error, no change
    assert mlflow.aliases["champion"] == "1"

    response = admin_client.post(url, {"reason": "better on every subgroup"})

    assert response.status_code == 302
    assert response["Location"] == f"/admin/registry/modelcandidate/{c.pk}/change/"
    assert mlflow.aliases["champion"] == "2"
    assert ModelApproval.objects.get().actor == "admin"


def test_rolling_back_through_the_admin(
    admin_client: Client, mlflow: FakeMlflow, events: list[ChampionChanged]
) -> None:
    promotion.approve(candidate().pk, reason="ok")
    response = admin_client.post("/admin/registry/modelcandidate/rollback/", {"reason": "bad"})
    assert response.status_code == 302
    assert mlflow.aliases == {"champion": "1"}


def test_staff_without_the_permission_cannot_decide(
    client: Client, mlflow: FakeMlflow, events: list[ChampionChanged]
) -> None:
    c = candidate()
    staff = User.objects.create_user("viewer", password="pw", is_staff=True)
    staff.user_permissions.add(Permission.objects.get(codename="view_modelcandidate"))
    client.force_login(staff)

    page = client.get(f"/admin/registry/modelcandidate/{c.pk}/change/")
    assert page.status_code == 200
    assert "/approve/" not in page.content.decode()
    response = client.post(f"/admin/registry/modelcandidate/{c.pk}/approve/", {"reason": "x"})
    assert response.status_code == 403
    assert mlflow.aliases["champion"] == "1"


def test_html_report_is_served_sandboxed(admin_client: Client, s3: FakeS3) -> None:
    c = candidate()
    s3.objects[("artifacts", "models/cxr-classifier/2/gate/vs-champion-v1.html")] = b"<h1>r</h1>"

    response = admin_client.get(f"/admin/registry/modelcandidate/{c.pk}/report/")

    assert response.status_code == 200
    assert response.content == b"<h1>r</h1>"
    assert "sandbox" in response["Content-Security-Policy"]


def test_html_report_outside_the_versions_gate_folder_is_refused(
    admin_client: Client, s3: FakeS3
) -> None:
    c = candidate(report_uri="s3://artifacts/models/cxr-classifier/9/model.onnx")
    assert admin_client.get(f"/admin/registry/modelcandidate/{c.pk}/report/").status_code == 404


# ---------------------------------------------------------------------- commands


def test_promote_command(mlflow: FakeMlflow, events: list[ChampionChanged]) -> None:
    candidate()
    with pytest.raises(CommandError, match="no candidate"):
        call_command("promote", "9", reason="x")
    call_command("promote", "2", reason="scripted")
    assert mlflow.aliases["champion"] == "2"
    assert ModelApproval.objects.get().actor == "cli"


def test_promote_auto_is_a_no_op_without_marked_candidates(
    mlflow: FakeMlflow, events: list[ChampionChanged], capsys: pytest.CaptureFixture[str]
) -> None:
    candidate()
    call_command("promote", auto=True)
    assert "no passed candidate" in capsys.readouterr().out
    assert mlflow.aliases["champion"] == "1"


def test_promote_auto_takes_a_given_version(
    mlflow: FakeMlflow, events: list[ChampionChanged]
) -> None:
    mlflow.versions["3"] = {}
    candidate("3", auto_promote=True)
    call_command("promote", "3", auto=True)
    assert mlflow.aliases["champion"] == "3"
    assert ModelApproval.objects.get().actor == "auto_promote"


def test_rollback_command(mlflow: FakeMlflow, events: list[ChampionChanged]) -> None:
    with pytest.raises(CommandError, match="nothing to roll back"):
        call_command("rollback", reason="x")
    promotion.approve(candidate().pk, reason="ok")
    call_command("rollback", reason="x", by="ops")
    assert mlflow.aliases["champion"] == "1"
