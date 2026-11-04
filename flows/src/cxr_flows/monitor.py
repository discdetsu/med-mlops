"""Prefect `monitor` flow: drift and delayed AUROC per model version, pushed to Prometheus.

    uv run python -m cxr_flows.monitor [run] [--config monitor|path.yaml]   # one pass
    uv run python -m cxr_flows.monitor serve [--config ...]                  # on a schedule

Each pass, for every model version with recent predictions or feedback (ADR 0008):

1. Its window, the newest `window_size` predictions of the last `max_window_age_minutes`, is
   compared with that version's own drift reference (`reference.parquet` in its bundle) with
   Evidently: embeddings, predicted probabilities, and metadata. A version's embeddings only
   mean something next to the same version's reference, so each version is judged against its
   own; the current `@champion` is the one serving almost all traffic. An unchanged window
   reuses its last report instead of writing a new one.
2. Delayed AUROC: its probabilities against the latest reviewer labels per job, with a
   bootstrap CI, reported only from `min_feedback` labeled predictions.
3. The HTML report goes to `s3://artifacts/monitoring/<model>/<version>/`, a row to
   `drift_reports` (Django admin lists them), and every score to the Pushgateway with
   `cxr_monitor_last_success_timestamp_seconds`. A failed pass pushes nothing, so the stale
   alert fires after three missed intervals.

It never starts a retrain: Phase 5 retrains on new labels, not on drift.
"""

import argparse
import json
import logging
import math
import os
import tempfile
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import boto3
import pandas as pd
from mlflow import MlflowClient
from prefect import flow, runtime, task
from prefect.cache_policies import NO_CACHE
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

from cxr_core.bundle import REFERENCE_FILE
from cxr_core.registry import (
    ARTIFACTS_BUCKET,
    CHAMPION_ALIAS,
    MODEL_NAME,
    bundle_uri,
    drift_report_prefix,
    split_s3_uri,
)
from cxr_flows import db, drift
from cxr_flows.config import MonitorConfig, load_monitor_config
from cxr_flows.registry import alias_version
from cxr_flows.settings import app_database_url, require_mlflow_tracking_uri

PUSH_JOB = "cxr_monitor"

# A cxr_flows logger: forwarded to the flow run's logs (cxr_flows/__init__.py), and usable
# when the functions below are called outside a flow (tests). Named explicitly: run with
# `python -m` this module is `__main__`, and `flow.serve` loads it by file path.
logger = logging.getLogger("cxr_flows.monitor")


@dataclass
class VersionReport:
    model_version: str
    is_champion: bool
    window_n: int
    scores: list[dict[str, Any]] = field(default_factory=list)  # empty: not judged
    drift_detected: bool = False
    report_uri: str = ""
    reused: bool = False
    skipped: str = ""  # why drift wasn't judged
    delayed_auroc: drift.DelayedAuroc | None = None


def reference_cache() -> Path:
    return Path(os.environ.get("MONITOR_CACHE_DIR") or tempfile.gettempdir()) / "cxr-references"


@task(cache_policy=NO_CACHE)
def load_reference(client: MlflowClient, version: str) -> pd.DataFrame:
    """The version's drift reference, downloaded once per version."""
    path = reference_cache() / MODEL_NAME / version / REFERENCE_FILE
    if not path.exists():
        mv = client.get_model_version(MODEL_NAME, version)
        bucket, prefix = split_s3_uri(mv.tags.get("bundle_uri") or bundle_uri(MODEL_NAME, version))
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".partial")
        boto3.client("s3").download_file(bucket, prefix + REFERENCE_FILE, str(partial))
        partial.rename(path)
    return pd.read_parquet(path)


@task(cache_policy=NO_CACHE)
def publish_report(
    result: drift.DriftResult, summary: dict[str, Any], version: str, stamp: str
) -> str:
    """Evidently HTML + our JSON summary under the version's monitoring prefix; returns the
    HTML's S3 URI."""
    key_prefix = f"{drift_report_prefix(MODEL_NAME, version)}{stamp}"
    s3 = boto3.client("s3")
    with tempfile.TemporaryDirectory() as tmp:
        html = result.save_html(Path(tmp) / "report.html")
        s3.upload_file(
            str(html), ARTIFACTS_BUCKET, f"{key_prefix}.html",
            {"ContentType": "text/html; charset=utf-8"},
        )  # fmt: skip
    s3.put_object(
        Bucket=ARTIFACTS_BUCKET,
        Key=f"{key_prefix}.json",
        Body=(json.dumps(summary, indent=2, default=str) + "\n").encode(),
        ContentType="application/json",
    )
    return f"s3://{ARTIFACTS_BUCKET}/{key_prefix}.html"


def monitor_version(
    *,
    cfg: MonitorConfig,
    url: str,
    client: MlflowClient,
    version: str,
    champion: str | None,
    rows: pd.DataFrame,
    feedback: pd.DataFrame,
    stamp: str,
    flow_run_id: str,
) -> VersionReport:
    report = VersionReport(version, version == champion, len(rows))
    reference = load_reference(client, version)
    report.delayed_auroc = drift.delayed_auroc(feedback, drift.classes_of(reference), cfg)

    if len(rows) < cfg.min_window_size:
        report.skipped = f"{len(rows)} recent predictions, at least {cfg.min_window_size} needed"
        return report

    fp = drift.fingerprint(rows["job_id"])
    last = db.latest_drift_report(url, MODEL_NAME, version)
    if last is not None and last["fingerprint"] == fp:
        report.scores = list(last["report"]["scores"])
        report.drift_detected = bool(last["drift"])
        report.report_uri, report.reused = last["report_uri"], True
        return report

    try:
        window = drift.window_frame(rows, reference)
        sample = drift.sample_reference(reference, cfg.reference_sample, cfg.seed)
        result = drift.compute_drift(sample, window, cfg)
    except drift.DriftError as exc:
        # A data problem with this version (e.g. predictions from a bundle whose embedding size
        # differs from its reference); the other versions are still judged.
        logger.error("version %s: drift not judged: %s", version, exc)
        report.skipped = str(exc)
        return report

    report.scores = [s.to_dict() for s in result.scores]
    report.drift_detected = result.detected
    window_start, window_end = rows["created_at"].min(), rows["created_at"].max()
    summary = {
        "model_name": MODEL_NAME,
        "model_version": version,
        "is_champion": report.is_champion,
        "window_start": window_start,
        "window_end": window_end,
        "n_predictions": result.n_window,
        "n_reference": result.n_reference,
        "scores": report.scores,
        "drift_detected": result.detected,
        "delayed_auroc": report.delayed_auroc.to_dict(),
        "config": cfg.model_dump(),
        "prefect_flow_run_id": flow_run_id,
    }
    report.report_uri = publish_report(result, summary, version, stamp)
    db.record_drift_report(
        url,
        model_name=MODEL_NAME,
        model_version=version,
        window_start=window_start,
        window_end=window_end,
        n_predictions=result.n_window,
        n_reference=result.n_reference,
        window_fingerprint=fp,
        drift_detected=result.detected,
        embedding_drift=result.max_score("embedding"),
        prediction_drift=result.max_score("prediction"),
        metadata_drift=result.max_score("metadata"),
        report=json.loads(json.dumps(summary, default=str)),
        report_uri=report.report_uri,
        prefect_flow_run_id=flow_run_id,
    )
    return report


def build_metrics(
    reports: list[VersionReport], cfg: MonitorConfig, champion: str | None, finished: float,
    duration_s: float,
) -> CollectorRegistry:  # fmt: skip
    """Everything one pass knows, as the gauges the alert rules and dashboards read."""
    reg = CollectorRegistry()
    score = Gauge("cxr_drift_score", "Drift score; above cxr_drift_threshold means drift",
                  ["model_version", "kind", "feature"], registry=reg)  # fmt: skip
    threshold = Gauge("cxr_drift_threshold", "Threshold for cxr_drift_score (monitor.yaml)",
                      ["model_version", "kind", "feature"], registry=reg)  # fmt: skip
    detected = Gauge("cxr_drift_detected", "1 when any score of the window is above threshold",
                     ["model_version"], registry=reg)  # fmt: skip
    window = Gauge("cxr_drift_window_predictions", "Predictions in the version's window",
                   ["model_version"], registry=reg)  # fmt: skip
    auroc = Gauge("cxr_delayed_auroc", "AUROC against reviewer labels (class_name=macro too)",
                  ["model_version", "class_name"], registry=reg)  # fmt: skip
    lower = Gauge("cxr_delayed_auroc_ci_lower", "Lower bound of the delayed AUROC CI",
                  ["model_version", "class_name"], registry=reg)  # fmt: skip
    upper = Gauge("cxr_delayed_auroc_ci_upper", "Upper bound of the delayed AUROC CI",
                  ["model_version", "class_name"], registry=reg)  # fmt: skip
    labeled = Gauge("cxr_delayed_auroc_labeled", "Labeled predictions behind the delayed AUROC",
                    ["model_version"], registry=reg)  # fmt: skip
    agree_share = Gauge("cxr_delayed_auroc_agree_share",
                        'Share of the labels that are "agree": the model scored against itself',
                        ["model_version"], registry=reg)  # fmt: skip
    Gauge("cxr_delayed_auroc_min_labeled", "Labeled predictions needed to report AUROC",
          registry=reg).set(cfg.min_feedback)  # fmt: skip
    if champion is not None:
        Gauge("cxr_monitor_champion", "1 for the @champion version at the last pass",
              ["model_version"], registry=reg).labels(champion).set(1)  # fmt: skip
    Gauge("cxr_monitor_interval_seconds", "Configured interval between passes",
          registry=reg).set(cfg.interval_seconds)  # fmt: skip
    Gauge("cxr_monitor_duration_seconds", "Duration of the last pass",
          registry=reg).set(duration_s)  # fmt: skip
    Gauge("cxr_monitor_last_success_timestamp_seconds", "When the last pass finished",
          registry=reg).set(finished)  # fmt: skip

    for r in reports:
        window.labels(r.model_version).set(r.window_n)
        if r.scores:
            detected.labels(r.model_version).set(int(r.drift_detected))
        for s in r.scores:
            labels = (r.model_version, s["kind"], s["feature"])
            score.labels(*labels).set(s["score"])
            threshold.labels(*labels).set(s["threshold"])
        perf = r.delayed_auroc
        if perf is None:
            continue
        labeled.labels(r.model_version).set(perf.n)
        if perf.agree_share is not None:
            agree_share.labels(r.model_version).set(perf.agree_share)
        if not perf.reported:
            continue
        rows: list[tuple[str, float | None, Any]] = [("macro", perf.macro, perf.macro_ci)]
        rows += [(c, v["auroc"], v["ci"]) for c, v in (perf.per_class or {}).items()]
        for name, value, ci in rows:
            if value is None or not math.isfinite(value):
                continue
            auroc.labels(r.model_version, name).set(value)
            if ci:
                lower.labels(r.model_version, name).set(ci[0])
                upper.labels(r.model_version, name).set(ci[1])
    return reg


@task(cache_policy=NO_CACHE)
def push_metrics(registry: CollectorRegistry) -> bool:
    url = os.environ.get("PUSHGATEWAY_URL")
    if not url:
        logger.warning(
            "PUSHGATEWAY_URL is not set: scores are not pushed (the Pushgateway is in `full`)"
        )
        return False
    # PUT replaces the whole group, so versions that left the window stop reporting scores.
    push_to_gateway(url, job=PUSH_JOB, registry=registry)
    return True


def _version_key(version: str) -> tuple[int, int, str]:
    """Numeric versions in numeric order, anything else after them."""
    return (0, int(version), "") if version.isdigit() else (1, 0, version)


@flow(name="monitor", log_prints=True)
def monitor_flow(config: str = "monitor") -> list[VersionReport]:
    started = time.monotonic()
    cfg = load_monitor_config(config)
    url = app_database_url()
    client = MlflowClient(require_mlflow_tracking_uri())
    now = datetime.now(UTC)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    flow_run_id = str(runtime.flow_run.id or "")

    champion = alias_version(client, MODEL_NAME, CHAMPION_ALIAS)
    windows = db.read_windows(
        url, MODEL_NAME, now - timedelta(minutes=cfg.max_window_age_minutes), cfg.window_size
    )
    feedback = db.read_feedback(url, MODEL_NAME, now - timedelta(days=cfg.feedback_window_days))
    versions = set(windows["model_version"]) | set(feedback["model_version"])
    if champion is not None:
        versions.add(champion)

    reports = [
        monitor_version(
            cfg=cfg,
            url=url,
            client=client,
            version=v,
            champion=champion,
            rows=windows[windows["model_version"] == v].reset_index(drop=True),
            feedback=feedback[feedback["model_version"] == v].reset_index(drop=True),
            stamp=stamp,
            flow_run_id=flow_run_id,
        )
        for v in sorted(versions, key=_version_key)
    ]
    for r in reports:
        _log_report(r)
    push_metrics(build_metrics(reports, cfg, champion, time.time(), time.monotonic() - started))
    return reports


def _log_report(r: VersionReport) -> None:
    tag = " (@champion)" if r.is_champion else ""
    if r.skipped:
        logger.info("version %s%s: drift not judged: %s", r.model_version, tag, r.skipped)
    else:
        drifted = [f"{s['kind']}:{s['feature']}={s['score']:.3f}" for s in r.scores if s["drifted"]]
        logger.info(
            "version %s%s: %d predictions, %s%s", r.model_version, tag, r.window_n,
            f"DRIFT in {', '.join(drifted)}" if drifted else "no drift",
            " (window unchanged, last report reused)" if r.reused else f" → {r.report_uri}",
        )  # fmt: skip
    perf = r.delayed_auroc
    if perf is not None and perf.n:
        if perf.reported and perf.macro is not None:
            ci = perf.macro_ci
            ci_text = f" (95% CI {ci[0]:.3f} to {ci[1]:.3f})" if ci else ""
            logger.info("version %s: delayed macro AUROC %.3f%s from %d labeled predictions, "
                        '%d of them "agree"',
                        r.model_version, perf.macro, ci_text, perf.n, perf.n_agree)  # fmt: skip
        else:
            logger.info("version %s: %d labeled predictions, AUROC needs %d", r.model_version,
                        perf.n, perf.min_n)  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", nargs="?", choices=("run", "serve"), default="run")
    parser.add_argument("--config", default="monitor", help="monitor config name or YAML path")
    args = parser.parse_args()
    if args.command == "serve":
        cfg = load_monitor_config(args.config)
        # Registers the `monitor/monitor` deployment with its schedule on the Prefect server and
        # runs each scheduled pass; the runner's webserver answers the compose healthcheck.
        monitor_flow.serve(
            name="monitor",
            interval=timedelta(seconds=cfg.interval_seconds),
            parameters={"config": args.config},
            webserver=True,
        )
    else:
        monitor_flow(config=args.config)


if __name__ == "__main__":
    main()
