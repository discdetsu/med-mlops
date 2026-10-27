import os
import uuid

import pandas as pd
import psycopg
import pytest

from cxr_flows.db import dataset_version, load_dataset_version
from cxr_flows.schema import CLASSES, COLUMNS

pytestmark = pytest.mark.integration


def _frame(prefix: str, split: str) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "image_id": [f"{prefix}_{i}" for i in range(3)],
            "file": [f"images/{prefix}_{i}.png" for i in range(3)],
            "patient_id": [f"{prefix}_p{i}" for i in range(3)],
            "patient_sex": ["M", "F", "M"],
            "patient_age": pd.array([30, None, 70], dtype="Int64"),
            "view_position": "PA",
            "follow_up": 0,
            "width": 1024,
            "height": 1024,
            "split": split,
            "label_source": "nih_nlp",
            **dict.fromkeys(CLASSES, 0),
        }
    )
    df.loc[0, CLASSES[0]] = 1
    return df.loc[:, list(COLUMNS)]


def test_load_is_idempotent_and_queryable() -> None:
    url = os.environ["APP_DATABASE_URL"]
    tag = f"_it{uuid.uuid4().hex[:8]}"
    rev = tag
    frames = {
        "data/nih": _frame(tag, "train"),
        "data/test_locked": _frame(tag + "t", "test_locked"),
    }
    md5s = {"data/nih": "a.dir", "data/test_locked": "b.dir"}
    try:
        for _ in range(2):
            load_dataset_version(url, rev, md5s, frames, "deadbeef", None)
        version = dataset_version(url, rev)
        assert version is not None
        assert version["counts"] == {"train": 3, "test_locked": 3}
        assert version["dvc_md5"] == md5s
        with psycopg.connect(url) as conn:
            images = conn.execute(
                "SELECT count(*), count(patient_age) FROM images WHERE dataset_rev = %s", (rev,)
            ).fetchone()
            positives = conn.execute(
                "SELECT count(*) FROM labels WHERE dataset_rev = %s AND value = 1", (rev,)
            ).fetchone()
        assert images == (6, 4)
        assert positives == (2,)
    finally:
        with psycopg.connect(url) as conn:
            conn.execute("DELETE FROM images WHERE dataset_rev = %s", (rev,))
            conn.execute("DELETE FROM dataset_versions WHERE dataset_rev = %s", (rev,))
