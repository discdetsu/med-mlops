"""The flows' rows in the app database.

Dataset metadata: `dataset_versions`, `images`, `dataset_images`, `labels`. The flows own these
tables and create them idempotently; Django will read them through unmanaged models when it
needs them (ADR 0006). The backend's tables (`inference_jobs`, `predictions`, `feedback`, see
ADR 0005) use explicit names too, so new tables here must not reuse those.

Promotion candidates: `model_candidates` is the exception. Django owns it (its migration
creates it when the backend starts) because the admin reads and updates it before and after
any gate runs; the gate flow only inserts rows (ADR 0007).

Monitoring: the monitor flow reads the backend's `predictions`, `inference_jobs`, and `feedback`,
and inserts into `drift_reports`, which Django owns for the same reason (ADR 0008).

Labeling: the monitor and export flows write `labeling_tasks` and `retrain_decisions`, again owned
by Django (ADR 0009).
"""

import json
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
import psycopg
from psycopg.types.json import Jsonb

from cxr_core.registry import (
    CANDIDATES_TABLE,
    DRIFT_REPORTS_TABLE,
    LABELING_TASKS_TABLE,
    RETRAIN_DECISIONS_TABLE,
    STATUS_GATE_FAILED,
    STATUS_PENDING,
)
from cxr_flows.schema import CLASSES

DDL = """
CREATE TABLE IF NOT EXISTS dataset_versions (
    dataset_rev         text PRIMARY KEY,
    dvc_md5             jsonb NOT NULL,       -- {"data/nih": "<md5>.dir", ...}
    counts              jsonb NOT NULL,       -- images per split
    classes             text[] NOT NULL,
    git_commit          text NOT NULL,
    prefect_flow_run_id text,
    created_at          timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS images (
    image_id      text PRIMARY KEY,
    patient_id    text NOT NULL,
    patient_sex   text NOT NULL,
    patient_age   integer,
    view_position text NOT NULL,
    follow_up     integer NOT NULL,
    width         integer NOT NULL,
    height        integer NOT NULL,
    split         text NOT NULL                 -- fixed per patient (ADR 0003)
);
CREATE INDEX IF NOT EXISTS images_patient_id ON images (patient_id);

-- Which images each version holds, so any version's membership can be rebuilt.
CREATE TABLE IF NOT EXISTS dataset_images (
    dataset_rev text NOT NULL REFERENCES dataset_versions ON DELETE CASCADE,
    dvc_path    text NOT NULL,                  -- data/nih, data/test_locked, data/curated
    image_id    text NOT NULL REFERENCES images,
    file        text NOT NULL,                  -- relative to dvc_path
    PRIMARY KEY (dataset_rev, dvc_path, image_id)
);

CREATE TABLE IF NOT EXISTS labels (
    image_id     text NOT NULL REFERENCES images ON DELETE CASCADE,
    label_source text NOT NULL,                 -- nih_nlp | reviewer
    class_name   text NOT NULL,
    value        smallint NOT NULL CHECK (value IN (0, 1)),
    PRIMARY KEY (image_id, label_source, class_name)
);
"""

_UPSERT_IMAGE = """
INSERT INTO images (image_id, patient_id, patient_sex, patient_age, view_position, follow_up,
                    width, height, split)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (image_id) DO UPDATE SET
    patient_id = EXCLUDED.patient_id, patient_sex = EXCLUDED.patient_sex,
    patient_age = EXCLUDED.patient_age, view_position = EXCLUDED.view_position,
    follow_up = EXCLUDED.follow_up, width = EXCLUDED.width, height = EXCLUDED.height,
    split = EXCLUDED.split
"""

_INSERT_MEMBER = """
INSERT INTO dataset_images (dataset_rev, dvc_path, image_id, file)
VALUES (%s, %s, %s, %s)
ON CONFLICT DO NOTHING
"""

_UPSERT_LABEL = """
INSERT INTO labels (image_id, label_source, class_name, value)
VALUES (%s, %s, %s, %s)
ON CONFLICT (image_id, label_source, class_name) DO UPDATE SET value = EXCLUDED.value
"""


def load_dataset_version(
    url: str,
    rev: str,
    md5s: Mapping[str, str],
    frames: Mapping[str, pd.DataFrame],
    git_commit: str,
    flow_run_id: str | None,
) -> None:
    """Record one dataset version, its images, and their labels in one transaction.

    `frames` maps a DVC path (e.g. `data/nih`) to its canonical metadata. Images and labels are
    upserted by `image_id`; `dataset_images` keeps every version's membership. Re-running with
    the same rev is a no-op apart from refreshing rows.
    """
    counts: dict[str, int] = {}
    for df in frames.values():
        for split, n in df["split"].value_counts().items():
            counts[str(split)] = counts.get(str(split), 0) + int(n)

    with psycopg.connect(url) as conn, conn.cursor() as cur:
        cur.execute(DDL)
        cur.execute(
            """
            INSERT INTO dataset_versions
                (dataset_rev, dvc_md5, counts, classes, git_commit, prefect_flow_run_id)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (dataset_rev) DO UPDATE SET
                counts = EXCLUDED.counts, git_commit = EXCLUDED.git_commit,
                prefect_flow_run_id = EXCLUDED.prefect_flow_run_id
            """,
            (rev, Jsonb(dict(md5s)), Jsonb(counts), list(CLASSES), git_commit, flow_run_id),
        )
        for dvc_path, df in frames.items():
            rows = _rows(df)
            cur.executemany(_UPSERT_IMAGE, [_image_row(r) for r in rows])
            cur.executemany(
                _INSERT_MEMBER, [(rev, dvc_path, r["image_id"], r["file"]) for r in rows]
            )
            cur.executemany(
                _UPSERT_LABEL,
                [(r["image_id"], r["label_source"], c, int(r[c])) for r in rows for c in CLASSES],
            )


def dataset_version(url: str, rev: str) -> dict[str, object] | None:
    with psycopg.connect(url) as conn, conn.cursor() as cur:
        cur.execute(DDL)
        cur.execute(
            "SELECT dvc_md5, counts, git_commit FROM dataset_versions WHERE dataset_rev = %s",
            (rev,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return {"dvc_md5": row[0], "counts": row[1], "git_commit": row[2]}


# Re-gating a version replaces its gate result and reopens the decision; earlier decisions
# stay in model_approvals.
_UPSERT_CANDIDATE = f"""
INSERT INTO {CANDIDATES_TABLE} (model_name, model_version, champion_version, status, gate_passed,
                                auto_promote, report, report_uri, mlflow_run_id, dataset_rev,
                                gated_at, decided_at)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now(), NULL)
ON CONFLICT (model_name, model_version) DO UPDATE SET
    champion_version = EXCLUDED.champion_version, status = EXCLUDED.status,
    gate_passed = EXCLUDED.gate_passed, auto_promote = EXCLUDED.auto_promote,
    report = EXCLUDED.report, report_uri = EXCLUDED.report_uri,
    mlflow_run_id = EXCLUDED.mlflow_run_id, dataset_rev = EXCLUDED.dataset_rev,
    gated_at = EXCLUDED.gated_at, decided_at = NULL
"""


def record_candidate(
    url: str,
    *,
    model_name: str,
    model_version: str,
    champion_version: str,
    gate_passed: bool,
    auto_promote: bool,
    report: Mapping[str, Any],
    report_uri: str,
    mlflow_run_id: str,
    dataset_rev: str,
) -> None:
    """Insert or refresh the version's row in `model_candidates`: `pending` approval when the
    gate passed, `gate_failed` otherwise."""
    status = STATUS_PENDING if gate_passed else STATUS_GATE_FAILED
    row = (
        model_name, model_version, champion_version, status, gate_passed, auto_promote,
        Jsonb(dict(report)), report_uri, mlflow_run_id, dataset_rev,
    )  # fmt: skip
    try:
        with psycopg.connect(url) as conn, conn.cursor() as cur:
            cur.execute(_UPSERT_CANDIDATE, row)
    except psycopg.errors.UndefinedTable:
        raise RuntimeError(
            f"{CANDIDATES_TABLE} doesn't exist yet: the backend's migrations create it when it "
            "starts (`make up`)"
        ) from None


# Each version's newest predictions since `since`, at most `limit` per version. The same file
# uploaded twice counts once (its newest prediction): it is one observation, and a duplicate on
# both sides of Evidently's train/test split lets the domain classifier memorize it, which reads
# as embedding drift. Rows from workers older than Phase 4 have no embedding and are left out.
_WINDOWS = """
SELECT job_id, model_version, created_at, probabilities, embedding,
       patient_sex, patient_age, view_position, patient_id, sha256, image_key
FROM (
    SELECT *, row_number() OVER (PARTITION BY model_version ORDER BY created_at DESC) AS rank
    FROM (
        SELECT DISTINCT ON (p.model_version, j.sha256)
               p.job_id, p.model_version, p.created_at, p.probabilities, p.embedding,
               j.patient_sex, j.patient_age, j.view_position, j.patient_id, j.sha256, p.image_key
        FROM predictions p JOIN inference_jobs j ON j.id = p.job_id
        WHERE p.model_name = %s AND p.created_at >= %s AND p.embedding IS NOT NULL
        ORDER BY p.model_version, j.sha256, p.created_at DESC
    ) unique_files
) newest
WHERE rank <= %s
ORDER BY model_version, created_at
"""

# The latest answer per job: a reviewer may correct their own feedback.
_FEEDBACK = """
SELECT DISTINCT ON (f.job_id)
       f.job_id, p.model_version, f.verdict, f.labels, p.probabilities, f.created_at,
       p.thresholds, j.patient_id, j.sha256, p.image_key
FROM feedback f
JOIN predictions p ON p.job_id = f.job_id
JOIN inference_jobs j ON j.id = f.job_id
WHERE p.model_name = %s AND f.created_at >= %s
ORDER BY f.job_id, f.created_at DESC, f.id DESC
"""

WINDOW_COLUMNS = (
    "job_id", "model_version", "created_at", "probabilities", "embedding",
    "patient_sex", "patient_age", "view_position", "patient_id", "sha256", "image_key",
)  # fmt: skip
FEEDBACK_COLUMNS = (
    "job_id", "model_version", "verdict", "labels", "probabilities", "created_at",
    "thresholds", "patient_id", "sha256", "image_key",
)  # fmt: skip


def read_windows(url: str, model_name: str, since: datetime, limit: int) -> pd.DataFrame:
    """Prediction windows for every version of `model_name` (see `_WINDOWS`)."""
    return _query(url, _WINDOWS, (model_name, since, limit), WINDOW_COLUMNS)


def read_feedback(url: str, model_name: str, since: datetime) -> pd.DataFrame:
    return _query(url, _FEEDBACK, (model_name, since), FEEDBACK_COLUMNS)


def latest_drift_report(url: str, model_name: str, version: str) -> dict[str, Any] | None:
    """The newest drift report row for the version, or None."""
    sql = f"""
        SELECT window_fingerprint, report, report_uri, drift_detected
        FROM {DRIFT_REPORTS_TABLE}
        WHERE model_name = %s AND model_version = %s
        ORDER BY created_at DESC, id DESC LIMIT 1
    """
    rows = _query(url, sql, (model_name, version), ("fingerprint", "report", "report_uri", "drift"))
    return None if rows.empty else {str(k): v for k, v in rows.iloc[0].items()}


_INSERT_DRIFT_REPORT = f"""
INSERT INTO {DRIFT_REPORTS_TABLE} (
    model_name, model_version, window_start, window_end, n_predictions, n_reference,
    window_fingerprint, drift_detected, embedding_drift, prediction_drift, metadata_drift,
    report, report_uri, prefect_flow_run_id, created_at)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
"""


def record_drift_report(
    url: str,
    *,
    model_name: str,
    model_version: str,
    window_start: datetime,
    window_end: datetime,
    n_predictions: int,
    n_reference: int,
    window_fingerprint: str,
    drift_detected: bool,
    embedding_drift: float | None,
    prediction_drift: float | None,
    metadata_drift: float | None,
    report: Mapping[str, Any],
    report_uri: str,
    prefect_flow_run_id: str,
) -> None:
    row = (
        model_name, model_version, window_start, window_end, n_predictions, n_reference,
        window_fingerprint, drift_detected, embedding_drift, prediction_drift, metadata_drift,
        Jsonb(dict(report)), report_uri, prefect_flow_run_id,
    )  # fmt: skip
    try:
        with psycopg.connect(url) as conn, conn.cursor() as cur:
            cur.execute(_INSERT_DRIFT_REPORT, row)
    except psycopg.errors.UndefinedTable:
        raise RuntimeError(_missing_backend_table(DRIFT_REPORTS_TABLE)) from None


# Any constant shared by every flow process; only has to be unique in the app database.
_LABELING_LOCK_ID = 0x6378726C  # "cxrl"


@contextmanager
def labeling_lock(url: str) -> Iterator[None]:
    """Serializes the labeling flows. The scheduled monitor, the scheduled export, and `make demo`
    may overlap; this keeps two of them from sending the same case to Label Studio, or from
    writing data/curated at once. Released when the connection closes, so a crash can't leave it
    held."""
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute("SELECT pg_advisory_lock(%s)", (_LABELING_LOCK_ID,))
        yield


def known_task_hashes(url: str, hashes: Iterable[str]) -> set[str]:
    """Which of these files already have a labeling task."""
    sql = f"SELECT sha256 FROM {LABELING_TASKS_TABLE} WHERE sha256 = ANY(%s)"
    rows = _query(url, sql, (list(hashes),), ("sha256",))
    return set(rows["sha256"])


_INSERT_TASK = f"""
INSERT INTO {LABELING_TASKS_TABLE} (sha256, job_id, model_name, model_version, reason, ls_task_id)
VALUES (%s, %s, %s, %s, %s, %s)
ON CONFLICT (sha256) DO NOTHING
"""


def record_tasks(url: str, rows: Iterable[tuple[str, str, str, str, str, int]]) -> None:
    """(sha256, job_id, model_name, model_version, reason, ls_task_id) rows."""
    try:
        with psycopg.connect(url) as conn, conn.cursor() as cur:
            cur.executemany(_INSERT_TASK, list(rows))
    except psycopg.errors.UndefinedTable:
        raise RuntimeError(_missing_backend_table(LABELING_TASKS_TABLE)) from None


@dataclass(frozen=True)
class TaskRow:
    """A labeling task with its job's upload details."""

    ls_task_id: int
    sha256: str
    status: str
    labels: dict[str, int] | None
    image_id: str
    filename: str
    format: str
    raw_key: str
    patient_id: str
    patient_sex: str
    patient_age: int | None
    view_position: str


def read_tasks(url: str) -> list[TaskRow]:
    sql = f"""
        SELECT t.ls_task_id, t.sha256, t.status, t.labels, t.image_id, j.filename, j.format,
               j.raw_key, j.patient_id, j.patient_sex, j.patient_age, j.view_position
        FROM {LABELING_TASKS_TABLE} t JOIN inference_jobs j ON j.id = t.job_id
        ORDER BY t.id
    """
    with psycopg.connect(url) as conn, conn.cursor() as cur:
        cur.execute(sql)
        return [TaskRow(*row) for row in cur.fetchall()]


def mark_exported(
    url: str, rows: Iterable[tuple[int, Mapping[str, int], str]], dataset_rev: str
) -> None:
    """(ls_task_id, labels, image_id) rows that are now in `dataset_rev`."""
    sql = f"""
        UPDATE {LABELING_TASKS_TABLE}
        SET status = 'exported', labels = %s, image_id = %s, dataset_rev = %s,
            quarantine_reason = '', updated_at = now()
        WHERE ls_task_id = %s
    """
    with psycopg.connect(url) as conn, conn.cursor() as cur:
        cur.executemany(
            sql, [(Jsonb(dict(labels)), image, dataset_rev, task) for task, labels, image in rows]
        )


def mark_quarantined(url: str, rows: Iterable[tuple[int, Mapping[str, int], str]]) -> None:
    """(ls_task_id, labels, reason) rows: the answer is kept here and goes no further."""
    sql = f"""
        UPDATE {LABELING_TASKS_TABLE}
        SET status = 'quarantined', labels = %s, quarantine_reason = %s, updated_at = now()
        WHERE ls_task_id = %s
    """
    with psycopg.connect(url) as conn, conn.cursor() as cur:
        cur.executemany(sql, [(Jsonb(dict(labels)), why, task) for task, labels, why in rows])


def task_counts(url: str) -> dict[str, int]:
    sql = f"SELECT status, count(*) FROM {LABELING_TASKS_TABLE} GROUP BY status"
    rows = _query(url, sql, (), ("status", "n"))
    return {str(s): int(n) for s, n in zip(rows["status"], rows["n"], strict=True)}


def candidate_status(url: str, model_name: str, version: str) -> dict[str, Any] | None:
    """The version's row in `model_candidates`, or None if it hasn't been gated."""
    sql = f"""
        SELECT status, gate_passed, report_uri FROM {CANDIDATES_TABLE}
        WHERE model_name = %s AND model_version = %s
    """
    rows = _query(url, sql, (model_name, version), ("status", "gate_passed", "report_uri"))
    return None if rows.empty else {str(k): v for k, v in rows.iloc[0].items()}


def curated_image_ids(url: str, rev: str) -> set[str]:
    """The data/curated images of a dataset version; empty for an unknown rev."""
    with psycopg.connect(url) as conn, conn.cursor() as cur:
        cur.execute(DDL)
        cur.execute(
            "SELECT image_id FROM dataset_images WHERE dataset_rev = %s AND dvc_path = %s",
            (rev, "data/curated"),
        )
        return {str(r[0]) for r in cur.fetchall()}


def latest_retrain_decision(url: str) -> dict[str, Any] | None:
    sql = f"""
        SELECT dataset_rev, new_labels, triggered, reason FROM {RETRAIN_DECISIONS_TABLE}
        ORDER BY created_at DESC, id DESC LIMIT 1
    """
    rows = _query(url, sql, (), ("dataset_rev", "new_labels", "triggered", "reason"))
    return None if rows.empty else {str(k): v for k, v in rows.iloc[0].items()}


def last_triggered_rev(url: str) -> str | None:
    """The dataset rev of the newest retrain that was started."""
    sql = f"""
        SELECT dataset_rev FROM {RETRAIN_DECISIONS_TABLE} WHERE triggered
        ORDER BY created_at DESC, id DESC LIMIT 1
    """
    rows = _query(url, sql, (), ("dataset_rev",))
    return None if rows.empty else str(rows.iloc[0]["dataset_rev"])


def record_retrain_decision(
    url: str,
    *,
    dataset_rev: str,
    baseline_rev: str,
    new_labels: int,
    min_new_labels: int,
    triggered: bool,
    reason: str,
    train_flow_run_id: str,
    prefect_flow_run_id: str,
) -> None:
    sql = f"""
        INSERT INTO {RETRAIN_DECISIONS_TABLE}
            (dataset_rev, baseline_rev, new_labels, min_new_labels, triggered, reason,
             train_flow_run_id, prefect_flow_run_id)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    row = (dataset_rev, baseline_rev, new_labels, min_new_labels, triggered, reason,
           train_flow_run_id, prefect_flow_run_id)  # fmt: skip
    try:
        with psycopg.connect(url) as conn, conn.cursor() as cur:
            cur.execute(sql, row)
    except psycopg.errors.UndefinedTable:
        raise RuntimeError(_missing_backend_table(RETRAIN_DECISIONS_TABLE)) from None


def _query(url: str, sql: str, params: tuple[Any, ...], columns: tuple[str, ...]) -> pd.DataFrame:
    try:
        with psycopg.connect(url) as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
    except psycopg.errors.UndefinedTable as exc:
        raise RuntimeError(_missing_backend_table(str(exc).split('"')[1])) from None
    return pd.DataFrame(rows, columns=list(columns))


def _missing_backend_table(table: str) -> str:
    return (
        f"{table} doesn't exist yet: the backend's migrations create it when it starts (`make up`)"
    )


def _rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    # Round-trip through JSON so pandas NA / numpy ints become plain None / int for psycopg.
    records: list[dict[str, Any]] = json.loads(df.to_json(orient="records"))
    return records


def _image_row(r: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        r["image_id"],
        r["patient_id"],
        r["patient_sex"],
        r["patient_age"],
        r["view_position"],
        r["follow_up"],
        r["width"],
        r["height"],
        r["split"],
    )
