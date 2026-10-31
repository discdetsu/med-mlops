"""The flows' rows in the app database.

Dataset metadata: `dataset_versions`, `images`, `dataset_images`, `labels`. The flows own these
tables and create them idempotently; Django will read them through unmanaged models when it
needs them (ADR 0006). The backend's tables (`inference_jobs`, `predictions`, `feedback`, see
ADR 0005) use explicit names too, so new tables here must not reuse those.

Promotion candidates: `model_candidates` is the exception. Django owns it (its migration
creates it when the backend starts) because the admin reads and updates it before and after
any gate runs; the gate flow only inserts rows (ADR 0007).
"""

import json
from collections.abc import Mapping
from typing import Any

import pandas as pd
import psycopg
from psycopg.types.json import Jsonb

from cxr_core.registry import CANDIDATES_TABLE, STATUS_GATE_FAILED, STATUS_PENDING
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
