"""Promotion in Django admin (ADR 0007): candidates with their gate report, approve / reject /
rollback behind a reason form, and the log of decisions.

Candidates are read-only here; only `registry.promotion` changes them. The buttons need the
`registry.promote_model` permission (superusers have it).
"""

import logging
from collections.abc import Callable
from typing import Any

from botocore.exceptions import BotoCoreError, ClientError
from django import forms
from django.contrib import admin, messages
from django.contrib.auth.models import User
from django.core.exceptions import PermissionDenied
from django.db.models import QuerySet
from django.db.models.fields.json import KT
from django.http import Http404, HttpRequest, HttpResponse, HttpResponseRedirect
from django.shortcuts import get_object_or_404
from django.template.response import TemplateResponse
from django.urls import URLPattern, path, reverse
from django.utils.html import format_html, format_html_join
from django.utils.safestring import SafeString
from mlflow.exceptions import MlflowException

from cxr_backend import storage
from cxr_backend.registry import promotion
from cxr_backend.registry.models import ModelApproval, ModelCandidate
from cxr_core.registry import ARTIFACTS_BUCKET, gate_report_prefix, split_s3_uri

logger = logging.getLogger(__name__)

PROMOTE_PERMISSION = "registry.promote_model"
# The report is our own escaped HTML, but it comes from object storage: no scripts, no requests.
REPORT_CSP = "default-src 'none'; style-src 'unsafe-inline'; sandbox"


_FIELDS = (
    "model_name",
    "model_version",
    "status",
    "champion_version",
    "gate_passed",
    "auto_promote",
    "gate_report",
    "report_link",
    "decisions",
    "mlflow_run_id",
    "dataset_rev",
    "gated_at",
    "decided_at",
)


class ReasonForm(forms.Form):
    reason = forms.CharField(
        widget=forms.Textarea(attrs={"rows": 4, "cols": 80}),
        help_text="Recorded with your username in model approvals.",
    )


@admin.register(ModelCandidate)
class ModelCandidateAdmin(admin.ModelAdmin):  # type: ignore[type-arg]
    list_display = [
        "model_version",
        "status",
        "gate_passed",
        "champion_version",
        "macro_delta",
        "gated_at",
        "decided_at",
    ]
    list_filter = ["status", "gate_passed"]
    search_fields = ["model_version", "dataset_rev", "mlflow_run_id"]
    fields = _FIELDS
    readonly_fields = _FIELDS
    change_form_template = "admin/registry/modelcandidate/change_form.html"
    change_list_template = "admin/registry/modelcandidate/change_list.html"

    # ------------------------------------------------------------------ read-only model

    def has_add_permission(self, request: HttpRequest, obj: object = None) -> bool:
        return False

    def has_change_permission(self, request: HttpRequest, obj: object = None) -> bool:
        return False

    def has_delete_permission(self, request: HttpRequest, obj: object = None) -> bool:
        return False

    def get_queryset(self, request: HttpRequest) -> QuerySet[ModelCandidate]:
        # The list needs one number from each report, not the whole report.
        qs: QuerySet[ModelCandidate] = (
            super()
            .get_queryset(request)
            .defer("report")
            .annotate(macro_delta_text=KT("report__macro_auroc__delta"))
        )
        return qs

    # ------------------------------------------------------------------ columns and fields

    @admin.display(description="macro ΔAUROC")
    def macro_delta(self, candidate: ModelCandidate) -> str:
        value = getattr(candidate, "macro_delta_text", None)
        return "-" if value in (None, "", "null") else f"{float(value):+.4f}"

    @admin.display(description="gate report")
    def gate_report(self, candidate: ModelCandidate) -> SafeString:
        return render_report(candidate.report)

    @admin.display(description="full report")
    def report_link(self, candidate: ModelCandidate) -> SafeString:
        if not candidate.report_uri:
            return format_html("{}", "-")
        url = reverse("admin:registry_modelcandidate_report", args=[candidate.pk])
        return format_html('<a href="{}" target="_blank" rel="noopener">HTML report</a>', url)

    @admin.display(description="decisions")
    def decisions(self, candidate: ModelCandidate) -> SafeString:
        rows = candidate.decisions.order_by("created_at")
        if not rows:
            return format_html("{}", "none yet")
        return format_html(
            "<ul>{}</ul>",
            format_html_join(
                "",
                "<li>{}: {} by {} ({} → {}): {}</li>",
                (
                    (
                        a.created_at.strftime("%Y-%m-%d %H:%M UTC"),
                        a.action,
                        a.actor,
                        a.from_version or "-",
                        a.to_version or "-",
                        a.reason,
                    )
                    for a in rows
                ),
            ),
        )

    # ------------------------------------------------------------------ pages

    def get_urls(self) -> list[URLPattern]:
        view = self.admin_site.admin_view
        custom = [
            path(
                "<int:pk>/approve/", view(self.approve_view), name="registry_modelcandidate_approve"
            ),
            path("<int:pk>/reject/", view(self.reject_view), name="registry_modelcandidate_reject"),
            path("<int:pk>/report/", view(self.report_view), name="registry_modelcandidate_report"),
            path("rollback/", view(self.rollback_view), name="registry_modelcandidate_rollback"),
        ]
        # Before the defaults, whose `<path:object_id>/` would match these first.
        return custom + super().get_urls()

    def changelist_view(
        self, request: HttpRequest, extra_context: dict[str, Any] | None = None
    ) -> HttpResponse:
        context = {**(extra_context or {}), **self._registry_context(request)}
        return super().changelist_view(request, context)

    def change_view(
        self,
        request: HttpRequest,
        object_id: str,
        form_url: str = "",
        extra_context: dict[str, Any] | None = None,
    ) -> HttpResponse:
        context = {**(extra_context or {}), **self._registry_context(request)}
        candidate = self.get_object(request, object_id)
        if candidate is not None and context["registry"] is not None:
            context["approval_blocker"] = promotion.approval_blocker(
                candidate, context["registry"].champion
            )
            context["can_reject"] = candidate.status in (
                ModelCandidate.Status.PENDING,
                ModelCandidate.Status.GATE_FAILED,
            )
        return super().change_view(request, object_id, form_url, context)

    def approve_view(self, request: HttpRequest, pk: int) -> HttpResponse:
        self._require_promote(request)
        candidate = get_object_or_404(ModelCandidate, pk=pk)
        state = self._state_or_none(request)
        champion = state.champion if state else None
        return self._decision_page(
            request,
            title=f"Approve version {candidate.model_version}",
            candidate=candidate,
            blocker=promotion.approval_blocker(candidate, champion)
            if state
            else "the model registry is unavailable",
            consequence=(
                f"@champion moves from version {champion} to version "
                f"{candidate.model_version}, version {champion} becomes @previous, and every "
                f"worker loads version {candidate.model_version} after its current job."
            ),
            submit="Approve and promote",
            act=lambda reason: promotion.approve(candidate.pk, reason=reason, user=_user(request)),
            done=f"Version {candidate.model_version} is now @champion.",
        )

    def reject_view(self, request: HttpRequest, pk: int) -> HttpResponse:
        self._require_promote(request)
        candidate = get_object_or_404(ModelCandidate, pk=pk)
        open_statuses = (ModelCandidate.Status.PENDING, ModelCandidate.Status.GATE_FAILED)
        return self._decision_page(
            request,
            title=f"Reject version {candidate.model_version}",
            candidate=candidate,
            blocker=None
            if candidate.status in open_statuses
            else f"it is already {candidate.get_status_display()}",
            consequence=(
                f"Version {candidate.model_version} is closed as rejected and loses @challenger "
                "if it holds it. @champion doesn't change."
            ),
            submit="Reject",
            act=lambda reason: promotion.reject(candidate.pk, reason=reason, user=_user(request)),
            done=f"Version {candidate.model_version} rejected.",
        )

    def rollback_view(self, request: HttpRequest) -> HttpResponse:
        self._require_promote(request)
        state = self._state_or_none(request)
        blocker = (
            promotion.rollback_blocker(state) if state else "the model registry is unavailable"
        )
        champion, previous = (state.champion, state.previous) if state else (None, None)
        return self._decision_page(
            request,
            title="Roll back @champion",
            candidate=None,
            blocker=blocker,
            consequence=(
                f"@champion moves from version {champion} back to version {previous}, "
                "@previous is cleared, and every worker loads version "
                f"{previous} after its current job."
            ),
            submit="Roll back",
            act=lambda reason: promotion.rollback(reason=reason, user=_user(request)),
            done=f"Rolled back: version {previous} is @champion again.",
        )

    def report_view(self, request: HttpRequest, pk: int) -> HttpResponse:
        candidate = get_object_or_404(ModelCandidate, pk=pk)
        if not self.has_view_permission(request, candidate):
            raise PermissionDenied
        try:
            bucket, key = split_s3_uri(candidate.report_uri)
        except ValueError:
            raise Http404("no report") from None
        expected = gate_report_prefix(candidate.model_name, candidate.model_version)
        if bucket != ARTIFACTS_BUCKET or not key.startswith(expected):
            raise Http404("report outside this version's gate folder")
        try:
            body = storage.get_object(bucket, key)
        except ClientError:
            raise Http404("report not found in object storage") from None
        response = HttpResponse(body, content_type="text/html; charset=utf-8")
        response["Content-Security-Policy"] = REPORT_CSP
        return response

    # ------------------------------------------------------------------ helpers

    def _decision_page(
        self,
        request: HttpRequest,
        *,
        title: str,
        candidate: ModelCandidate | None,
        blocker: str | None,
        consequence: str,
        submit: str,
        act: Callable[[str], promotion.Decision],
        done: str,
    ) -> HttpResponse:
        back = (
            reverse("admin:registry_modelcandidate_change", args=[candidate.pk])
            if candidate
            else reverse("admin:registry_modelcandidate_changelist")
        )
        form = ReasonForm(request.POST if request.method == "POST" else None)
        if request.method == "POST" and blocker is None and form.is_valid():
            try:
                decision = act(form.cleaned_data["reason"])
            except promotion.PromotionError as exc:
                self.message_user(request, str(exc), messages.ERROR)
            except (MlflowException, OSError, BotoCoreError) as exc:
                logger.exception("promotion action failed")
                self.message_user(request, f"Nothing changed: {exc}", messages.ERROR)
            else:
                self.message_user(request, done, messages.SUCCESS)
                if decision.approval.to_version != decision.approval.from_version and (
                    not decision.event_published
                ):
                    self.message_user(
                        request,
                        "Workers weren't notified (RabbitMQ unavailable). They load the new "
                        "@champion when they reconnect; `docker compose restart worker` forces it.",
                        messages.WARNING,
                    )
            return HttpResponseRedirect(back)
        context = {
            **self.admin_site.each_context(request),
            "opts": self.opts,
            "title": title,
            "candidate": candidate,
            "blocker": blocker,
            "consequence": consequence,
            "submit": submit,
            "form": form,
            "back": back,
        }
        return TemplateResponse(request, "admin/registry/decision.html", context)

    def _require_promote(self, request: HttpRequest) -> None:
        if not request.user.has_perm(PROMOTE_PERMISSION):
            raise PermissionDenied

    def _state_or_none(self, request: HttpRequest) -> promotion.RegistryState | None:
        try:
            return promotion.registry_state()
        except MlflowException as exc:
            logger.warning("registry state unavailable: %s", exc)
            return None

    def _registry_context(self, request: HttpRequest) -> dict[str, Any]:
        state = self._state_or_none(request)
        return {
            "registry": state,
            "can_promote": request.user.has_perm(PROMOTE_PERMISSION),
            "can_roll_back": state is not None and promotion.rollback_blocker(state) is None,
        }


@admin.register(ModelApproval)
class ModelApprovalAdmin(admin.ModelAdmin):  # type: ignore[type-arg]
    list_display = [
        "created_at",
        "action",
        "from_version",
        "to_version",
        "actor",
        "event_published",
        "candidate",
    ]
    list_filter = ["action", "event_published"]
    search_fields = ["actor", "reason", "from_version", "to_version"]
    readonly_fields = [f.name for f in ModelApproval._meta.fields]

    def has_add_permission(self, request: HttpRequest, obj: object = None) -> bool:
        return False

    def has_change_permission(self, request: HttpRequest, obj: object = None) -> bool:
        return False

    def has_delete_permission(self, request: HttpRequest, obj: object = None) -> bool:
        return False


def _user(request: HttpRequest) -> User:
    """The signed-in admin user; admin views only run for authenticated staff."""
    user = request.user
    if not isinstance(user, User):
        raise PermissionDenied
    return user


def render_report(report: dict[str, Any]) -> SafeString:
    """The gate report (cxr_flows.gating.decide) as admin HTML; every value is escaped."""
    if not report:
        return format_html("{}", "no report")
    pct = f"{report.get('config', {}).get('confidence', 0.95):.0%}"
    checks = format_html_join(
        "",
        '<tr><td>{}</td><td style="color:{}"><strong>{}</strong></td><td>{}</td></tr>',
        (
            (c["name"], *(("#1b5e20", "pass") if c["passed"] else ("#b00020", "FAIL")), c["detail"])
            for c in report.get("checks", [])
        ),
    )
    macro = report.get("macro_auroc", {})
    auroc_rows = format_html_join(
        "",
        "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
        [
            ("macro", _f(macro.get("champion")), _f(macro.get("challenger")), _delta(macro)),
            *(
                (v.get("class"), _f(v.get("champion")), _f(v.get("challenger")), _delta(v))
                for v in report.get("classes", [])
            ),
        ],
    )
    subgroup_rows = format_html_join(
        "",
        "<tr><td>{}={}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
        (
            (
                s.get("attribute"),
                s.get("value"),
                s.get("n"),
                _f(s.get("champion")),
                _f(s.get("challenger")),
                _delta(s),
                ("pass" if s.get("passed") else "FAIL") if s.get("judged") else "not judged",
            )
            for s in report.get("subgroups", [])
        ),
    )
    ece = report.get("ece", {})
    verdict = "PASSED" if report.get("passed") else "FAILED"
    return format_html(
        "<p><strong>Gate {}</strong>: version {} vs @champion version {} on {} locked-test "
        "images from {} patients. Δ = challenger - champion, {} CIs from a paired bootstrap "
        "over patients.</p>"
        "<table><thead><tr><th>Check</th><th>Result</th><th>Detail</th></tr></thead>"
        "<tbody>{}</tbody></table>"
        "<h3>AUROC</h3><table><thead><tr><th>Class</th><th>Champion</th><th>Challenger</th>"
        "<th>Δ ({} CI)</th></tr></thead><tbody>{}</tbody></table>"
        "<h3>Subgroups (macro AUROC)</h3><table><thead><tr><th>Group</th><th>n</th>"
        "<th>Champion</th><th>Challenger</th><th>Δ ({} CI)</th><th>Result</th></tr></thead>"
        "<tbody>{}</tbody></table>"
        "<p>Pooled ECE: champion {}, challenger {}.</p>",
        verdict,
        report.get("challenger_version"),
        report.get("champion_version"),
        report.get("n"),
        report.get("n_patients"),
        pct,
        checks,
        pct,
        auroc_rows,
        pct,
        subgroup_rows,
        _f(ece.get("champion")),
        _f(ece.get("challenger")),
    )


def _f(value: float | None, sign: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.4f}" if sign else f"{value:.4f}"


def _delta(entry: dict[str, Any]) -> str:
    ci = entry.get("delta_ci")
    interval = f" ({_f(ci[0], True)} to {_f(ci[1], True)})" if ci else ""
    return _f(entry.get("delta"), sign=True) + interval
