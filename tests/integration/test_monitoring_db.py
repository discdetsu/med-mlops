"""The monitor flow reads the backend's tables and inserts into `drift_reports` with SQL, while
Django's migrations own all of them (ADR 0008). These check the queries against the real
Postgres schema: the backend must have started (`make up`), since its migrations create them.

The end-to-end check (replay → inject drift → alert; stop the monitor → stale alert) is manual;
see the README's "Monitoring" section.
"""

import os
import uuid
from datetime import UTC, datetime, timedelta

import psycopg
import pytest
from psycopg.types.json import Jsonb

from cxr_flows import db

pytestmark = pytest.mark.integration

CLASSES = ["Effusion", "Mass"]


def _job(cur: psycopg.Cursor, model: str, version: str, sha: str, *, at: datetime,
         embedding: list[float] | None, view: str = "PA") -> uuid.UUID:  # fmt: skip
    job_id = uuid.uuid4()
    cur.execute(
        """INSERT INTO inference_jobs (id, status, filename, format, size_bytes, sha256, raw_key,
                                       patient_sex, patient_age, view_position, error, created_at)
           VALUES (%s, 'succeeded', 'x.png', 'png', 1, %s, 'raw/x.png', 'F', 61, %s, '', %s)""",
        (job_id, sha, view, at),
    )
    cur.execute(
        """INSERT INTO predictions (job_id, model_name, model_version, preprocess_version,
                                    mlflow_run_id, dataset_rev, classes, probabilities, thresholds,
                                    image_key, heatmap_keys, latency_ms, worker, embedding,
                                    created_at)
           VALUES (%s, %s, %s, '1', '', '', %s, %s, %s, 'k', %s, 1.0, 'w', %s, %s)""",
        (
            job_id, model, version, Jsonb(CLASSES), Jsonb(dict.fromkeys(CLASSES, 0.4)),
            Jsonb(dict.fromkeys(CLASSES, 0.5)), Jsonb(dict.fromkeys(CLASSES, "k")),
            None if embedding is None else Jsonb(embedding), at,
        ),
    )  # fmt: skip
    return job_id


def test_windows_feedback_and_drift_reports_fit_the_backend_tables() -> None:
    url = os.environ["APP_DATABASE_URL"]
    model = f"_it{uuid.uuid4().hex[:8]}"
    now = datetime.now(UTC)
    try:
        with psycopg.connect(url) as conn, conn.cursor() as cur:
            # The same file twice: only its newer prediction counts.
            _job(cur, model, "3", "a" * 64, at=now - timedelta(minutes=5), embedding=[0.0, 1.0])
            newer = _job(cur, model, "3", "a" * 64, at=now - timedelta(minutes=1),
                         embedding=[0.5, 1.0])  # fmt: skip
            other = _job(cur, model, "3", "b" * 64, at=now - timedelta(minutes=2),
                         embedding=[1.0, 1.0], view="AP")  # fmt: skip
            _job(cur, model, "3", "c" * 64, at=now - timedelta(days=2), embedding=[2.0, 2.0])
            _job(cur, model, "3", "d" * 64, at=now, embedding=None)  # pre-Phase 4 worker
            _job(cur, model, "4", "a" * 64, at=now, embedding=[9.0, 9.0])
            # Two answers on one job: the later one counts.
            for verdict, mass in (("agree", 0), ("correct", 1)):
                cur.execute(
                    "INSERT INTO feedback (job_id, model_version, verdict, labels, comment, "
                    "created_at) VALUES (%s, '3', %s, %s, '', now())",
                    (newer, verdict, Jsonb({"Effusion": 0, "Mass": mass})),
                )

        windows = db.read_windows(url, model, now - timedelta(hours=1), limit=10)
        v3 = windows[windows["model_version"] == "3"]
        assert list(v3["job_id"]) == [other, newer]  # oldest first, deduplicated, no NULLs
        assert v3.iloc[1]["embedding"] == [0.5, 1.0]
        assert v3.iloc[0]["view_position"] == "AP" and v3.iloc[0]["patient_age"] == 61
        assert set(windows["model_version"]) == {"3", "4"}  # same file, other version: kept
        assert len(db.read_windows(url, model, now - timedelta(hours=1), limit=1)) == 2

        feedback = db.read_feedback(url, model, now - timedelta(days=1))
        assert len(feedback) == 1
        assert feedback.iloc[0]["labels"] == {"Effusion": 0, "Mass": 1}
        assert feedback.iloc[0]["probabilities"] == {"Effusion": 0.4, "Mass": 0.4}

        assert db.latest_drift_report(url, model, "3") is None
        report = {"scores": [{"kind": "embedding", "score": 0.7}], "delayed_auroc": None}
        db.record_drift_report(
            url, model_name=model, model_version="3", window_start=now, window_end=now,
            n_predictions=2, n_reference=10, window_fingerprint="fp", drift_detected=True,
            embedding_drift=0.7, prediction_drift=None, metadata_drift=0.1, report=report,
            report_uri="s3://artifacts/monitoring/x/3/t.html", prefect_flow_run_id="",
        )  # fmt: skip
        latest = db.latest_drift_report(url, model, "3")
        assert latest == {
            "fingerprint": "fp", "report": report, "drift": True,
            "report_uri": "s3://artifacts/monitoring/x/3/t.html",
        }  # fmt: skip
    finally:
        # Django cascades in Python, not in the schema: children first.
        with psycopg.connect(url) as conn:
            conn.execute("DELETE FROM drift_reports WHERE model_name = %s", (model,))
            jobs = [r[0] for r in conn.execute(
                "SELECT job_id FROM predictions WHERE model_name = %s", (model,)
            )]  # fmt: skip
            conn.execute("DELETE FROM feedback WHERE job_id = ANY(%s)", (jobs,))
            conn.execute("DELETE FROM predictions WHERE job_id = ANY(%s)", (jobs,))
            conn.execute("DELETE FROM inference_jobs WHERE id = ANY(%s)", (jobs,))
