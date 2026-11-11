"""The labeling flows write `labeling_tasks` and `retrain_decisions` and read the backend's tables
with SQL, while Django's migrations own them all (ADR 0009). These check the queries against the
real Postgres schema: the backend must have started (`make up`), since its migrations create them.

The end-to-end check, including Label Studio, is `make demo` under the `full` profile.
"""

import os
import uuid
from datetime import UTC, datetime, timedelta

import pandas as pd
import psycopg
import pytest
from psycopg.types.json import Jsonb

from cxr_flows import db
from cxr_flows.schema import CLASSES, COLUMNS

pytestmark = pytest.mark.integration

PROBS = dict.fromkeys(CLASSES, 0.4)


def _job(cur: psycopg.Cursor, model: str, sha: str, *, patient: str = "") -> uuid.UUID:
    job_id = uuid.uuid4()
    now = datetime.now(UTC)
    cur.execute(
        """INSERT INTO inference_jobs (id, status, filename, format, size_bytes, sha256, raw_key,
                                       patient_sex, patient_age, view_position, patient_id, error,
                                       created_at)
           VALUES (%s, 'succeeded', 'x.png', 'png', 1, %s, %s, 'F', 61, 'AP', %s, '', %s)""",
        (job_id, sha, f"raw/{job_id}.png", patient, now),
    )
    cur.execute(
        """INSERT INTO predictions (job_id, model_name, model_version, preprocess_version,
                                    mlflow_run_id, dataset_rev, classes, probabilities, thresholds,
                                    image_key, heatmap_keys, latency_ms, worker, embedding,
                                    created_at)
           VALUES (%s, %s, '3', '1', '', '', %s, %s, %s, %s, %s, 1.0, 'w', %s, %s)""",
        (
            job_id, model, Jsonb(list(CLASSES)), Jsonb(PROBS), Jsonb(dict.fromkeys(CLASSES, 0.5)),
            f"results/{job_id}/image.png", Jsonb(dict.fromkeys(CLASSES, "k")), Jsonb([0.0, 1.0]),
            now,
        ),
    )  # fmt: skip
    return job_id


def test_tasks_fit_the_backend_tables() -> None:
    url = os.environ["APP_DATABASE_URL"]
    model = f"_it{uuid.uuid4().hex[:8]}"
    a, b = uuid.uuid4().hex, uuid.uuid4().hex
    base = hash(model) % 1_000_000 + 900_000_000
    now = datetime.now(UTC)
    try:
        with psycopg.connect(url) as conn, conn.cursor() as cur:
            job_a = _job(cur, model, a, patient="00000002")
            job_b = _job(cur, model, b)
            cur.execute(
                "INSERT INTO feedback (job_id, model_version, verdict, labels, comment, created_at)"
                " VALUES (%s, '3', 'correct', %s, '', now())",
                (job_a, Jsonb(dict.fromkeys(CLASSES, 1))),
            )

        # The columns selection needs, on both reads.
        windows = db.read_windows(url, model, now - timedelta(hours=1), limit=10)
        row = windows[windows["job_id"] == job_a].iloc[0]
        assert (row["sha256"], row["patient_id"]) == (a, "00000002")
        assert row["image_key"] == f"results/{job_a}/image.png"
        feedback = db.read_feedback(url, model, now - timedelta(hours=1))
        assert feedback.iloc[0]["thresholds"] == dict.fromkeys(CLASSES, 0.5)
        assert (feedback.iloc[0]["sha256"], feedback.iloc[0]["patient_id"]) == (a, "00000002")

        assert db.known_task_hashes(url, {a, b}) == set()
        db.record_tasks(
            url,
            [
                (a, str(job_a), model, "3", "disagreement", base),
                (b, str(job_b), model, "3", "random", base + 1),
            ],
        )
        # The same file again: the first task stands.
        db.record_tasks(url, [(a, str(job_a), model, "3", "drift", base + 2)])
        assert db.known_task_hashes(url, {a, b, "other"}) == {a, b}

        rows = {t.ls_task_id: t for t in db.read_tasks(url) if t.sha256 in (a, b)}
        assert set(rows) == {base, base + 1}
        first = rows[base]
        assert (first.status, first.format, first.patient_id) == ("open", "png", "00000002")
        assert (first.patient_sex, first.patient_age, first.view_position) == ("F", 61, "AP")
        assert first.raw_key == f"raw/{job_a}.png" and first.labels is None

        labels = dict.fromkeys(CLASSES, 0) | {"Mass": 1}
        db.mark_exported(url, [(base, labels, "upload-a")], "rev1")
        db.mark_quarantined(url, [(base + 1, labels, "patient is in test_locked")])
        rows = {t.ls_task_id: t for t in db.read_tasks(url) if t.sha256 in (a, b)}
        assert (rows[base].status, rows[base].image_id) == ("exported", "upload-a")
        assert rows[base].labels == labels
        assert rows[base + 1].status == "quarantined"
        with psycopg.connect(url) as conn:
            reason = conn.execute(
                "SELECT quarantine_reason FROM labeling_tasks WHERE ls_task_id = %s", (base + 1,)
            ).fetchone()
        assert reason == ("patient is in test_locked",)
        counts = db.task_counts(url)
        assert counts["exported"] >= 1 and counts["quarantined"] >= 1
    finally:
        with psycopg.connect(url) as conn:
            jobs = [r[0] for r in conn.execute(
                "SELECT job_id FROM predictions WHERE model_name = %s", (model,)
            )]  # fmt: skip
            conn.execute("DELETE FROM labeling_tasks WHERE job_id = ANY(%s)", (jobs,))
            conn.execute("DELETE FROM feedback WHERE job_id = ANY(%s)", (jobs,))
            conn.execute("DELETE FROM predictions WHERE job_id = ANY(%s)", (jobs,))
            conn.execute("DELETE FROM inference_jobs WHERE id = ANY(%s)", (jobs,))


def test_retrain_decisions_fit_their_table() -> None:
    url = os.environ["APP_DATABASE_URL"]
    rev = f"_it{uuid.uuid4().hex[:8]}"
    try:
        db.record_retrain_decision(
            url, dataset_rev=rev, baseline_rev="base", new_labels=3, min_new_labels=50,
            triggered=False, reason="3 new labels, at least 50 needed", train_flow_run_id="",
            prefect_flow_run_id="run",
        )  # fmt: skip
        assert db.latest_retrain_decision(url) == {
            "dataset_rev": rev, "new_labels": 3, "triggered": False,
            "reason": "3 new labels, at least 50 needed",
        }  # fmt: skip
        db.record_retrain_decision(
            url, dataset_rev=rev, baseline_rev="base", new_labels=60, min_new_labels=50,
            triggered=True, reason="yes", train_flow_run_id="t", prefect_flow_run_id="run",
        )  # fmt: skip
        assert db.last_triggered_rev(url) == rev
        assert db.last_train_run_id(url) == "t"
    finally:
        with psycopg.connect(url) as conn:
            conn.execute("DELETE FROM retrain_decisions WHERE dataset_rev = %s", (rev,))


def test_a_curated_row_fits_the_dataset_tables() -> None:
    """Reviewer labels and a rev holding data/curated go through the same load as an ingest."""
    url = os.environ["APP_DATABASE_URL"]
    rev = f"_it{uuid.uuid4().hex[:8]}"
    image = f"upload-{uuid.uuid4().hex[:16]}"
    row = {
        "image_id": image, "file": f"images/{image}.png", "patient_id": "00000002",
        "patient_sex": "F", "patient_age": 61, "view_position": "AP", "follow_up": 0,
        "width": 1024, "height": 1024, "split": "recent", "label_source": "reviewer",
        **dict.fromkeys(CLASSES, 0), "Mass": 1,
    }  # fmt: skip
    frame = pd.DataFrame([row], columns=list(COLUMNS))
    try:
        db.load_dataset_version(
            url, rev, {"data/curated": "x.dir"}, {"data/curated": frame}, "c", ""
        )
        assert db.curated_image_ids(url, rev) == {image}
        assert db.curated_image_ids(url, "unknown-rev") == set()
        with psycopg.connect(url) as conn:
            mass = conn.execute(
                "SELECT value FROM labels WHERE image_id = %s AND label_source = 'reviewer' "
                "AND class_name = 'Mass'",
                (image,),
            ).fetchone()
        assert mass == (1,)
    finally:
        with psycopg.connect(url) as conn:
            conn.execute("DELETE FROM dataset_versions WHERE dataset_rev = %s", (rev,))
            conn.execute("DELETE FROM images WHERE image_id = %s", (image,))
