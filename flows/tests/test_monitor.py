"""The monitor flow's per-version step and its Prometheus output, with storage faked."""

from typing import Any

import numpy as np
import pandas as pd
import pytest
from prometheus_client import CollectorRegistry, generate_latest

from cxr_flows import monitor
from cxr_flows.config import MonitorConfig, load_monitor_config
from cxr_flows.drift import DelayedAuroc

from .test_drift import _feedback, _population, _rows


@pytest.fixture
def cfg() -> MonitorConfig:
    return load_monitor_config().model_copy(
        update={"min_window_size": 80, "min_feedback": 20, "bootstrap_samples": 100}
    )


class FakeDb:
    def __init__(self) -> None:
        self.reports: list[dict[str, Any]] = []

    def latest_drift_report(self, url: str, model_name: str, version: str) -> Any:
        mine = [r for r in self.reports if r["model_version"] == version]
        if not mine:
            return None
        r = mine[-1]
        return {
            "fingerprint": r["window_fingerprint"],
            "report": r["report"],
            "report_uri": r["report_uri"],
            "drift": r["drift_detected"],
        }

    def record_drift_report(self, url: str, **row: Any) -> None:
        self.reports.append(row)


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> tuple[FakeDb, list[str]]:
    db = FakeDb()
    published: list[str] = []
    reference = _population(1000, np.random.default_rng(0), ap_share=0.37)
    monkeypatch.setattr(monitor.db, "latest_drift_report", db.latest_drift_report)
    monkeypatch.setattr(monitor.db, "record_drift_report", db.record_drift_report)
    monkeypatch.setattr(monitor, "load_reference", lambda client, version: reference)

    def publish(result: Any, summary: dict[str, Any], version: str, stamp: str) -> str:
        uri = f"s3://artifacts/monitoring/cxr-classifier/{version}/{stamp}.html"
        published.append(uri)
        return uri

    monkeypatch.setattr(monitor, "publish_report", publish)
    return db, published


def _run(cfg: MonitorConfig, rows: pd.DataFrame, feedback: pd.DataFrame | None = None,
         stamp: str = "t1") -> monitor.VersionReport:  # fmt: skip
    return monitor.monitor_version(
        cfg=cfg, url="db", client=None, version="2", champion="2", rows=rows,  # type: ignore[arg-type]
        feedback=feedback if feedback is not None else pd.DataFrame(columns=["labels"]),
        stamp=stamp, flow_run_id="run-1",
    )  # fmt: skip


def test_drifted_window_is_reported_recorded_and_published(
    cfg: MonitorConfig, fakes: tuple[FakeDb, list[str]]
) -> None:
    db, published = fakes
    rows = _rows(_population(120, np.random.default_rng(1), ap_share=1.0))
    report = _run(cfg, rows, _feedback(25, np.random.default_rng(2), 3))

    assert report.drift_detected and report.is_champion and not report.reused
    assert report.report_uri == published[0]
    [row] = db.reports
    assert row["n_predictions"] == 120
    assert row["drift_detected"] is True
    assert row["embedding_drift"] > cfg.embedding_threshold
    assert row["report"]["delayed_auroc"]["reported"] is True
    assert row["report"]["scores"] == report.scores


def test_unchanged_window_reuses_the_last_report(
    cfg: MonitorConfig, fakes: tuple[FakeDb, list[str]]
) -> None:
    db, published = fakes
    rows = _rows(_population(100, np.random.default_rng(1), ap_share=0.37))
    first = _run(cfg, rows)
    again = _run(cfg, rows, stamp="t2")

    assert again.reused and again.scores == first.scores
    assert again.report_uri == first.report_uri
    assert len(db.reports) == 1 and len(published) == 1


def test_small_window_is_not_judged_but_feedback_still_is(
    cfg: MonitorConfig, fakes: tuple[FakeDb, list[str]]
) -> None:
    db, _ = fakes
    rows = _rows(_population(10, np.random.default_rng(1), ap_share=1.0))
    report = _run(cfg, rows, _feedback(30, np.random.default_rng(2), 3))

    assert report.scores == [] and "at least 80" in report.skipped
    assert report.delayed_auroc is not None and report.delayed_auroc.reported
    assert db.reports == []


def test_mismatched_embeddings_skip_the_version(
    cfg: MonitorConfig, fakes: tuple[FakeDb, list[str]]
) -> None:
    rows = _rows(_population(100, np.random.default_rng(1), ap_share=0.37))
    rows["embedding"] = [[0.0] * 3] * len(rows)
    report = _run(cfg, rows)
    assert report.scores == [] and "3 values" in report.skipped


def _metrics(reg: CollectorRegistry) -> str:
    return generate_latest(reg).decode()


def test_metrics_carry_scores_thresholds_auroc_and_the_heartbeat(cfg: MonitorConfig) -> None:
    judged = monitor.VersionReport(
        "2", True, 150,
        scores=[{"kind": "embedding", "feature": "embedding", "score": 0.71, "threshold": 0.6,
                 "drifted": True}],
        drift_detected=True,
        delayed_auroc=DelayedAuroc(40, 20, macro=0.8, macro_ci=(0.7, 0.9),
                                   per_class={"Mass": {"auroc": 0.75, "ci": [0.6, 0.9]},
                                              "Effusion": {"auroc": None, "ci": None}}),
    )  # fmt: skip
    quiet = monitor.VersionReport("1", False, 12, delayed_auroc=DelayedAuroc(5, 20))
    text = _metrics(monitor.build_metrics([judged, quiet], cfg, "2", 1_000.0, 2.5))

    for line in (
        'cxr_drift_score{feature="embedding",kind="embedding",model_version="2"} 0.71',
        'cxr_drift_threshold{feature="embedding",kind="embedding",model_version="2"} 0.6',
        'cxr_drift_detected{model_version="2"} 1.0',
        'cxr_drift_window_predictions{model_version="1"} 12.0',
        'cxr_delayed_auroc{class_name="macro",model_version="2"} 0.8',
        'cxr_delayed_auroc_ci_lower{class_name="Mass",model_version="2"} 0.6',
        'cxr_delayed_auroc_labeled{model_version="1"} 5.0',
        'cxr_monitor_champion{model_version="2"} 1.0',
        "cxr_monitor_interval_seconds 300.0",
        "cxr_monitor_last_success_timestamp_seconds 1000.0",
    ):
        assert line in text, line
    # Not judged → no drift series for version 1; undefined AUROC → no series.
    assert 'cxr_drift_detected{model_version="1"}' not in text
    assert 'class_name="Effusion"' not in text
    assert 'cxr_delayed_auroc{class_name="macro",model_version="1"}' not in text


def test_push_is_skipped_without_a_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PUSHGATEWAY_URL", raising=False)
    assert monitor.push_metrics.fn(CollectorRegistry()) is False


def test_versions_sort_numerically() -> None:
    assert sorted(["10", "2", "1"], key=monitor._version_key) == ["1", "2", "10"]
