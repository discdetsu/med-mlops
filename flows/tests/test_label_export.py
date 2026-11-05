"""The export flow's data handling, with Label Studio, S3, DVC, and the database faked."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandera.errors
import pytest
from botocore.exceptions import ClientError
from PIL import Image

from cxr_core.registry import TASK_EXPORTED, TASK_OPEN, TASK_QUARANTINED
from cxr_flows import db, label_export, labeling
from cxr_flows.labelstudio import NONE_CHOICE, choices_result
from cxr_flows.nih import patient_split
from cxr_flows.schema import (
    CLASSES,
    partition_labels,
    read_metadata,
    validate_curated,
    write_metadata,
)
from cxr_flows.settings import Paths

from .conftest import make_rows


def _patient(split: str, curated: str | None = None, start: int = 0) -> str:
    for i in range(start, start + 100_000):
        p = f"{i:08d}"
        if patient_split(p) == split and (curated is None or labeling.curated_split(p) == curated):
            return p
    raise AssertionError(split, curated)


TRAIN_PATIENT = _patient("train", "train")
RECENT_PATIENT = _patient("train", "recent")
VAL_PATIENT = _patient("val")
LOCKED_PATIENT = _patient("test_locked")


def _png(path: Path, size: tuple[int, int] = (256, 256)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.random.default_rng(0).integers(0, 256, (size[1], size[0]), dtype=np.uint8)
    Image.fromarray(pixels).save(path)


def _task(ls_id: int, sha: str, **fields: Any) -> db.TaskRow:
    values = {
        "ls_task_id": ls_id,
        "sha256": sha * 64,
        "status": TASK_OPEN,
        "labels": None,
        "image_id": "",
        "filename": "x.png",
        "format": "png",
        "raw_key": f"raw/{sha}.png",
        "patient_id": TRAIN_PATIENT,
        "patient_sex": "F",
        "patient_age": 61,
        "view_position": "AP",
    }
    return db.TaskRow(**(values | fields))


class FakeLs:
    def __init__(self, answers: dict[int, list[str]]) -> None:
        self.answers = answers

    def list_tasks(self, project: int) -> list[dict[str, Any]]:
        return [
            {
                "id": task_id,
                "annotations": [
                    {"id": 1, "result": choices_result(choices), "was_cancelled": False}
                ],
            }
            for task_id, choices in self.answers.items()
        ]


class FakeS3:
    def __init__(
        self, missing: set[str] = frozenset(), sizes: dict[str, tuple[int, int]] | None = None
    ) -> None:  # type: ignore[assignment]
        self.missing = missing
        self.sizes = sizes or {}
        self.downloads: list[str] = []

    def download_file(self, bucket: str, key: str, dest: str) -> None:
        self.downloads.append(key)
        if key in self.missing:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        _png(Path(dest), self.sizes.get(key, (256, 256)))


@pytest.fixture
def repo(tmp_path: Path) -> Paths:
    paths = Paths(tmp_path)
    locked = make_rows(2, split="test_locked")
    locked["patient_id"] = LOCKED_PATIENT
    paths.test_locked.mkdir(parents=True)
    write_metadata(locked, paths.test_locked)
    nih = make_rows(2)
    paths.nih.mkdir(parents=True)
    write_metadata(nih, paths.nih)
    return paths


class Harness:
    """Runs `export_answers`, with `dvc add` writing a pointer the way the real one would."""

    def __init__(self, paths: Paths, monkeypatch: pytest.MonkeyPatch) -> None:
        self.paths = paths
        self.tasks: list[db.TaskRow] = []
        self.exported: list[Any] = []
        self.quarantined: list[Any] = []
        self.versions: list[dict[str, Any]] = []
        self.s3 = FakeS3()
        self.pushed = 0
        monkeypatch.setattr(label_export.db, "read_tasks", lambda url: list(self.tasks))
        monkeypatch.setattr(
            label_export.db,
            "mark_exported",
            lambda url, rows, rev: self.exported.append((rows, rev)),
        )
        monkeypatch.setattr(
            label_export.db, "mark_quarantined", lambda url, rows: self.quarantined.extend(rows)
        )

        def record(url: str, rev: str, md5s: Any, frames: Any, commit: str, run: str) -> None:
            self.versions.append({"rev": rev, "md5s": md5s, "frames": frames})

        monkeypatch.setattr(label_export.db, "load_dataset_version", record)
        monkeypatch.setattr(label_export.boto3, "client", lambda name: self.s3)
        monkeypatch.setattr(label_export, "ensure_pinned", lambda root, targets: None)
        self.clean = True
        self.restored = 0
        monkeypatch.setattr(label_export, "is_clean", lambda root, target: self.clean)

        def restore(root: Path, target: Path) -> None:
            self.restored += 1

        monkeypatch.setattr(label_export, "dvc_restore", restore)
        monkeypatch.setattr(label_export, "git_commit", lambda root: "abc")

        def add(root: Path, targets: list[Path]) -> None:
            assert targets == [paths.curated]
            digest = _digest(paths.curated / "metadata.csv")
            (paths.data / "curated.dvc").write_text(
                f"outs:\n- md5: {digest}.dir\n  path: curated\n"
            )

        def push(root: Path, targets: list[Path]) -> None:
            self.pushed += 1

        monkeypatch.setattr(label_export, "dvc_add", add)
        monkeypatch.setattr(label_export, "dvc_push", push)
        for name, target in (("nih", paths.nih), ("test_locked", paths.test_locked)):
            (paths.data / f"{name}.dvc").write_text(
                f"outs:\n- md5: {name}.dir\n  path: {target.name}\n"
            )

    def run(self, answers: dict[int, list[str]], push: bool = True) -> label_export.ExportResult:
        return label_export.export_answers.fn(self.paths, "db", FakeLs(answers), 1, push)


def _digest(path: Path) -> str:
    import hashlib

    return hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()


@pytest.fixture
def harness(repo: Paths, monkeypatch: pytest.MonkeyPatch) -> Harness:
    return Harness(repo, monkeypatch)


def test_answers_become_curated_rows_a_version_and_exported_tasks(harness: Harness) -> None:
    harness.tasks = [_task(1, "a"), _task(2, "b", patient_id=RECENT_PATIENT), _task(3, "c")]
    result = harness.run({1: ["Mass"], 2: [NONE_CHOICE]})  # task 3 has no answer yet

    assert result.exported == 2 and result.quarantined == 0
    curated = read_metadata(harness.paths.curated).set_index("image_id")
    assert list(curated.index) == ["upload-" + "a" * 16, "upload-" + "b" * 16]
    a, b = curated.iloc[0], curated.iloc[1]
    assert (a["split"], a["label_source"], a["Mass"], a["Effusion"]) == ("train", "reviewer", 1, 0)
    assert (b["split"], b["Mass"]) == ("recent", 0)
    assert (harness.paths.curated / a["file"]).exists()
    assert (a["width"], a["height"], a["patient_id"]) == (256, 256, TRAIN_PATIENT)

    version = harness.versions[0]
    assert version["rev"] == result.dataset_rev
    assert set(version["frames"]) == {"data/nih", "data/test_locked", "data/curated"}
    assert set(version["md5s"]) == {"data/nih", "data/test_locked", "data/curated"}
    rows, rev = harness.exported[0]
    assert rev == result.dataset_rev
    assert sorted(r[0] for r in rows) == [1, 2]
    assert harness.pushed == 1


def test_the_curated_set_passes_the_check_training_runs(harness: Harness) -> None:
    harness.tasks = [_task(1, "a"), _task(2, "b", patient_id=VAL_PATIENT)]
    harness.run({1: ["Mass"], 2: ["Mass"]})
    curated = read_metadata(harness.paths.curated)
    validate_curated(
        curated, harness.paths.curated, read_metadata(harness.paths.nih),
        read_metadata(harness.paths.test_locked),
    )  # fmt: skip
    assert curated.set_index("patient_id").at[VAL_PATIENT, "split"] == "recent"


def test_test_locked_patients_are_quarantined_and_never_reach_the_data(harness: Harness) -> None:
    harness.tasks = [_task(1, "a"), _task(2, "b", patient_id=LOCKED_PATIENT)]
    result = harness.run({1: ["Mass"], 2: ["Mass"]})

    assert (result.exported, result.quarantined) == (1, 1)
    [(task_id, labels, why)] = harness.quarantined
    assert task_id == 2 and labels["Mass"] == 1
    assert "test_locked" in why
    curated = read_metadata(harness.paths.curated)
    assert LOCKED_PATIENT not in set(curated["patient_id"])
    assert not (harness.paths.curated / "images" / f"upload-{'b' * 16}.png").exists()
    frames = harness.versions[0]["frames"]["data/curated"]
    assert LOCKED_PATIENT not in set(frames["patient_id"])


def test_a_patient_only_in_the_locked_list_is_quarantined_too(harness: Harness) -> None:
    """The hash bucket says train, but the frozen set is the authority (it predates the hash)."""
    locked = read_metadata(harness.paths.test_locked)
    locked["patient_id"] = TRAIN_PATIENT
    write_metadata(locked, harness.paths.test_locked)
    harness.tasks = [_task(1, "a")]
    result = harness.run({1: ["Mass"]})
    assert result.quarantined == 1 and result.dataset_rev is None
    assert "patient is in test_locked" in harness.quarantined[0][2]


@pytest.mark.parametrize(
    ("fields", "expect"),
    [
        ({"patient_sex": ""}, "patient_sex"),
        ({"view_position": ""}, "view_position"),
        ({"view_position": "LATERAL"}, "view_position"),
    ],
)
def test_unusable_metadata_is_quarantined_not_guessed(
    harness: Harness, fields: dict[str, str], expect: str
) -> None:
    harness.tasks = [_task(1, "a", **fields), _task(2, "b")]
    result = harness.run({1: ["Mass"], 2: ["Mass"]})
    assert (result.exported, result.quarantined) == (1, 1)
    assert expect in harness.quarantined[0][2]


def test_an_expired_upload_is_quarantined(harness: Harness) -> None:
    harness.s3 = FakeS3(missing={"raw/a.png"})
    harness.tasks = [_task(1, "a"), _task(2, "b")]
    result = harness.run({1: ["Mass"], 2: ["Mass"]})
    assert (result.exported, result.quarantined) == (1, 1)
    assert harness.quarantined[0][0] == 1 and "NoSuchKey" in harness.quarantined[0][2]


def test_a_too_small_image_is_quarantined(harness: Harness) -> None:
    harness.s3 = FakeS3(sizes={"raw/a.png": (100, 100)})
    harness.tasks = [_task(1, "a")]
    result = harness.run({1: ["Mass"]})
    assert result.quarantined == 1 and "width" in harness.quarantined[0][2]


def test_nothing_answered_changes_nothing(harness: Harness) -> None:
    harness.tasks = [_task(1, "a")]
    result = harness.run({})
    assert result == label_export.ExportResult(None, 0, 0, None)
    assert not harness.paths.curated.exists() and not harness.versions


def test_exported_tasks_are_not_processed_again(harness: Harness) -> None:
    harness.tasks = [_task(1, "a")]
    first = harness.run({1: ["Mass"]})
    harness.s3.downloads.clear()
    harness.tasks = [
        _task(
            1,
            "a",
            status=TASK_EXPORTED,
            labels={
                "Effusion": 0,
                "Mass": 1,
                "Atelectasis": 0,
                "Cardiomegaly": 0,
                "Pneumothorax": 0,
            },
        )
    ]
    again = harness.run({1: ["Mass"]})
    assert again.dataset_rev == first.dataset_rev and again.exported == 0
    assert harness.s3.downloads == [] and len(harness.versions) == 1


def test_a_changed_answer_updates_its_row_and_makes_a_new_rev(harness: Harness) -> None:
    harness.tasks = [_task(1, "a")]
    first = harness.run({1: ["Mass"]})
    labels = dict.fromkeys(CLASSES, 0) | {"Mass": 1}
    harness.tasks = [_task(1, "a", status=TASK_EXPORTED, labels=labels)]
    second = harness.run({1: ["Effusion"]})

    curated = read_metadata(harness.paths.curated)
    assert (curated.at[0, "Mass"], curated.at[0, "Effusion"]) == (0, 1)
    assert second.dataset_rev != first.dataset_rev
    assert second.exported == 0  # no new label towards a retrain
    assert harness.s3.downloads == ["raw/a.png"]  # the file is already there
    rows, _ = harness.exported[-1]
    assert rows[0][0] == 1 and rows[0][1]["Effusion"] == 1


def test_quarantined_tasks_stay_quarantined(harness: Harness) -> None:
    harness.tasks = [_task(1, "a", status=TASK_QUARANTINED)]
    assert harness.run({1: ["Mass"]}).exported == 0
    assert harness.s3.downloads == []


def test_rerun_after_a_crash_gives_the_same_rev(harness: Harness) -> None:
    """The tasks are still open if the process died after `dvc add`: identical content, same rev."""
    harness.tasks = [_task(1, "a"), _task(2, "b")]
    first = harness.run({1: ["Mass"], 2: ["Mass"]})
    second = harness.run({1: ["Mass"], 2: ["Mass"]})
    assert first.dataset_rev == second.dataset_rev
    assert len(read_metadata(harness.paths.curated)) == 2


def test_a_write_interrupted_before_dvc_add_is_restored_first(harness: Harness) -> None:
    harness.tasks = [_task(1, "a")]
    harness.run({1: ["Mass"]})
    assert harness.restored == 0
    harness.clean = False  # the process died after writing files, before `dvc add`
    harness.run({1: ["Mass"]})
    assert harness.restored == 1


def test_labels_accumulate_across_exports(harness: Harness) -> None:
    harness.tasks = [_task(1, "a")]
    harness.run({1: ["Mass"]})
    harness.tasks = [_task(2, "b")]
    result = harness.run({2: [NONE_CHOICE]})
    assert result.exported == 1
    assert len(read_metadata(harness.paths.curated)) == 2


def test_no_push_skips_the_remote(harness: Harness) -> None:
    harness.tasks = [_task(1, "a")]
    harness.run({1: ["Mass"]}, push=False)
    assert harness.pushed == 0


def test_partition_keeps_rows_that_pass_as_a_valid_frame(harness: Harness, tmp_path: Path) -> None:
    rows = make_rows(1, split="train").assign(label_source="reviewer")
    _png(tmp_path / rows.at[0, "file"])
    accepted, refused = partition_labels(rows, [], tmp_path)
    assert refused == {} and len(accepted) == 1


def test_partition_raises_on_a_broken_frame_instead_of_quarantining(tmp_path: Path) -> None:
    rows = make_rows(1, split="train").assign(label_source="reviewer").drop(columns=["follow_up"])
    with pytest.raises((ValueError, pandera.errors.SchemaErrors)):
        partition_labels(rows, [], tmp_path)


def test_decision_counts_only_new_training_images(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness.tasks = [
        _task(1, "a"), _task(2, "b"), _task(3, "c", patient_id=RECENT_PATIENT),
    ]  # fmt: skip
    result = harness.run({1: ["Mass"], 2: ["Mass"], 3: ["Mass"]})

    class Version:
        tags = {"dataset_rev": "base"}  # noqa: RUF012

    monkeypatch.setattr(label_export, "newest_version", lambda c, m: Version)
    monkeypatch.setattr(label_export, "MlflowClient", lambda uri: None)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://x")
    monkeypatch.setattr(
        label_export.db, "curated_image_ids", lambda url, rev: {"upload-" + "a" * 16}
    )
    monkeypatch.setattr(label_export.db, "last_triggered_rev", lambda url: None)
    assert result.dataset_rev is not None
    decision = label_export.decide_retrain.fn(harness.paths, "db", result.dataset_rev, 1)
    # a: the baseline has it; c: a recent patient, which never trains. Only b is new.
    assert decision.new_labels == 1 and decision.trigger and decision.baseline_rev == "base"


def test_a_scheduled_pass_with_no_change_adds_no_decision_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows: list[dict[str, Any]] = []
    decision = labeling.should_retrain(
        dataset_rev="r", baseline_rev="b", last_triggered_rev=None, new_labels=3, min_new_labels=9
    )

    def latest(url: str) -> dict[str, Any] | None:
        return dict(rows[-1]) if rows else None

    monkeypatch.setattr(label_export.db, "latest_retrain_decision", latest)
    monkeypatch.setattr(label_export.db, "record_retrain_decision", lambda url, **r: rows.append(r))
    monkeypatch.setattr(
        label_export, "runtime", type("R", (), {"flow_run": type("F", (), {"id": "x"})})
    )
    label_export._record_decision("db", decision, True, "", None)
    label_export._record_decision("db", decision, True, "", None)
    assert len(rows) == 1
    moved = labeling.should_retrain(
        dataset_rev="r", baseline_rev="b", last_triggered_rev=None, new_labels=4, min_new_labels=9
    )
    label_export._record_decision("db", moved, True, "", None)
    assert len(rows) == 2


def test_metrics_report_tasks_images_and_progress() -> None:
    from prometheus_client import generate_latest

    curated = pd.DataFrame({"split": ["train", "train", "recent"]})
    decision = labeling.should_retrain(
        dataset_rev="r", baseline_rev="b", last_triggered_rev=None, new_labels=3, min_new_labels=9
    )
    text = generate_latest(
        label_export.build_metrics({"open": 4, "exported": 2}, curated, decision)
    ).decode()
    for line in (
        'cxr_labeling_tasks{status="open"} 4.0',
        'cxr_labeling_tasks{status="quarantined"} 0.0',
        'cxr_curated_images{split="train"} 2.0',
        'cxr_curated_images{split="recent"} 1.0',
        "cxr_retrain_new_labels 3.0",
        "cxr_retrain_min_new_labels 9.0",
    ):
        assert line in text, line
