"""Drift reports in Django admin: read-only rows written by the monitor flow, each with a link
to its Evidently HTML report, streamed from object storage."""

from botocore.exceptions import ClientError
from django.contrib import admin
from django.core.exceptions import PermissionDenied
from django.http import Http404, HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404
from django.urls import URLPattern, path, reverse
from django.utils.html import format_html, format_html_join
from django.utils.safestring import SafeString

from cxr_backend import storage
from cxr_backend.monitoring.models import DriftReport
from cxr_core.registry import ARTIFACTS_BUCKET, drift_report_prefix, split_s3_uri

# Evidently's report is a self-contained React page: it needs inline scripts and styles, but
# never eval or the network (checked in headless Chromium). `sandbox allow-scripts` without
# `allow-same-origin` runs it in an opaque origin, so it can't read the admin's cookies or DOM.
REPORT_CSP = (
    "default-src 'none'; script-src 'unsafe-inline'; "
    "style-src 'unsafe-inline'; img-src data: blob:; font-src data:; sandbox allow-scripts"
)


@admin.register(DriftReport)
class DriftReportAdmin(admin.ModelAdmin):  # type: ignore[type-arg]
    list_display = [
        "created_at",
        "model_version",
        "n_predictions",
        "drift_detected",
        "embedding_drift",
        "prediction_drift",
        "metadata_drift",
        "report_link",
    ]
    list_filter = ["model_version", "drift_detected"]
    readonly_fields = [
        *(f.name for f in DriftReport._meta.fields if f.name != "report"),
        "scores",
        "delayed_auroc",
        "report_link",
    ]
    exclude = ["report"]

    @admin.display(description="report")
    def report_link(self, report: DriftReport) -> SafeString:
        if not report.report_uri:
            return format_html("{}", "-")
        url = reverse("admin:monitoring_driftreport_report", args=[report.pk])
        return format_html('<a href="{}" target="_blank" rel="noopener">Evidently</a>', url)

    @admin.display(description="scores")
    def scores(self, report: DriftReport) -> SafeString:
        rows = report.report.get("scores", [])
        return format_html(
            "<table><tr><th>kind</th><th>feature</th><th>method</th><th>n</th><th>score</th>"
            "<th>threshold</th><th>drift</th></tr>{}</table>",
            format_html_join(
                "",
                "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td>"
                "<td>{}</td></tr>",
                (
                    (
                        r["kind"],
                        r["feature"],
                        r["method"],
                        r["n"],
                        f"{r['score']:.3f}",
                        f"{r['threshold']:.3f}",
                        "yes" if r["drifted"] else "no",
                    )
                    for r in rows
                ),
            ),
        )

    @admin.display(description="delayed AUROC")
    def delayed_auroc(self, report: DriftReport) -> SafeString:
        perf = report.report.get("delayed_auroc")
        if not perf:
            return format_html("{}", "-")
        if not perf.get("reported"):
            return format_html(
                "not reported: {} labeled predictions, at least {} needed",
                perf["n"],
                perf["min_n"],
            )
        ci = perf["macro_ci"]
        ci_text = f" (95% CI {ci[0]:.3f} to {ci[1]:.3f})" if ci else ""
        return format_html("macro {}{}, n = {}", f"{perf['macro']:.3f}", ci_text, perf["n"])

    def get_urls(self) -> list[URLPattern]:
        view = self.admin_site.admin_view
        custom = [
            path(
                "<int:pk>/report/",
                view(self.report_view),
                name="monitoring_driftreport_report",
            )
        ]
        return custom + super().get_urls()

    def report_view(self, request: HttpRequest, pk: int) -> HttpResponse:
        report = get_object_or_404(DriftReport, pk=pk)
        if not self.has_view_permission(request, report):
            raise PermissionDenied
        try:
            bucket, key = split_s3_uri(report.report_uri)
        except ValueError:
            raise Http404("no report") from None
        expected = drift_report_prefix(report.model_name, report.model_version)
        if bucket != ARTIFACTS_BUCKET or not key.startswith(expected):
            raise Http404("report outside this version's monitoring folder")
        try:
            body = storage.get_object(bucket, key)
        except ClientError:
            raise Http404("report not found in object storage") from None
        response = HttpResponse(body, content_type="text/html; charset=utf-8")
        response["Content-Security-Policy"] = REPORT_CSP
        return response

    def has_add_permission(self, request: HttpRequest, obj: object = None) -> bool:
        return False

    def has_change_permission(self, request: HttpRequest, obj: object = None) -> bool:
        return False
