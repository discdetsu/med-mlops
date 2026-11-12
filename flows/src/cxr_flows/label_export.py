"""Prefect `label-export` flow: finished annotations in Label Studio → `labels` (reviewer) →
`data/curated/` → `dvc add` + `dvc push` → `dataset_versions` → `should_retrain()`.

    uv run python -m cxr_flows.label_export [--config labeling|path.yaml] [--no-push]
                                            [--no-trigger] [--auto-promote]
                                            [--min-new-labels N]

Runs every `export_interval_seconds` under the `flows-worker` service, or once with
`make label-export`.

Every answer is checked with Pandera before it can enter the data. A row that fails, such as a
patient in test_locked, is quarantined on its `labeling_tasks` row and never reaches `labels`,
`images`, or `data/curated/`; the others carry on. A label that a reviewer later changes updates
its row and makes a new rev, but isn't a new label: `should_retrain()` still counts only images
the newest model version didn't have.

While a retrain this flow started is unfinished, an export writes nothing and decides nothing, so
it can't change the data under the run or queue a second one.
"""

import argparse
import json
import logging
import os
import shutil
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import boto3
import pandas as pd
from botocore.exceptions import ClientError
from mlflow import MlflowClient
from prefect import flow, get_client, runtime, task
from prefect.cache_policies import NO_CACHE
from prefect.deployments import run_deployment
from prefect.exceptions import ObjectNotFound
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

from cxr_core.registry import (
    INFERENCE_BUCKET,
    MODEL_NAME,
    TASK_EXPORTED,
    TASK_OPEN,
    TASK_QUARANTINED,
)
from cxr_flows import db, labeling
from cxr_flows.config import LabelingConfig, load_labeling_config
from cxr_flows.labelstudio import LabelStudio, ensure_review_project
from cxr_flows.registry import newest_version
from cxr_flows.schema import (
    CLASSES,
    COLUMNS,
    CURATED_RECENT,
    CURATED_TRAIN,
    image_size,
    partition_labels,
    read_metadata,
    write_metadata,
)
from cxr_flows.settings import Paths, app_database_url, require_mlflow_tracking_uri
from cxr_flows.versioning import dataset_rev as rev_of
from cxr_flows.versioning import (
    dvc_add,
    dvc_push,
    dvc_restore,
    ensure_pinned,
    git_commit,
    is_clean,
    pinned_md5s,
)

PUSH_JOB = "cxr_labeling"
_EXTENSIONS = {"png": ".png", "jpeg": ".jpg", "dicom": ".dcm"}
# What S3 answers for an object that isn't there: boto's download does a HEAD first, hence "404".
_NO_SUCH_OBJECT = {"404", "NoSuchKey", "NotFound"}
# Named explicitly: run with `python -m` this module is `__main__`.
logger = logging.getLogger("cxr_flows.label_export")


class RetrainNotStarted(RuntimeError):
    """should_retrain() said yes, but the deployment couldn't be started."""


@dataclass(frozen=True)
class ExportResult:
    dataset_rev: str | None  # None until the first label is exported
    exported: int
    quarantined: int
    decision: labeling.RetrainDecision | None
    train_flow_run_id: str = ""


def curated_row(row: db.TaskRow, labels: dict[str, int], staging: Path) -> dict[str, Any]:
    """A `labeling_tasks` row + its answer → a canonical metadata row, with the file in `staging`.

    Sizes come from the file, so a missing one fails the schema check. Unknown sex or view stay
    blank, which fails it too: subgroup metrics need them.
    """
    image_id = labeling.image_id_for(row.sha256)
    file = f"images/{image_id}{_EXTENSIONS[row.format]}"
    size = image_size(staging / file) or (0, 0)
    return {
        "image_id": image_id,
        "file": file,
        "patient_id": row.patient_id,
        "patient_sex": row.patient_sex,
        "patient_age": pd.NA if row.patient_age is None else row.patient_age,
        "view_position": row.view_position,
        "follow_up": 0,  # uploads don't carry it; nothing reads it for curated images
        "width": size[0],
        "height": size[1],
        "split": labeling.curated_split(row.patient_id),
        "label_source": "reviewer",
        **labels,
    }


def as_metadata(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=list(COLUMNS))
    frame["patient_age"] = frame["patient_age"].astype("Int64")
    for column in ("follow_up", "width", "height", *CLASSES):
        frame[column] = frame[column].astype("int64")
    return frame


def apply_labels(
    existing: pd.DataFrame | None, accepted: pd.DataFrame, changed: dict[str, dict[str, int]]
) -> pd.DataFrame:
    """data/curated's new metadata: `existing`, with labels updated for `changed` image IDs and
    `accepted` rows added or replacing theirs. Sorted, so equal content hashes equally."""
    base = existing if existing is not None else accepted.iloc[:0]
    base = base[~base["image_id"].isin(accepted["image_id"])].copy()
    for image_id, labels in changed.items():
        for c, value in labels.items():
            base.loc[base["image_id"] == image_id, c] = value
    frames = [f for f in (base, accepted) if not f.empty]
    if not frames:
        return accepted.iloc[:0]
    return pd.concat(frames, ignore_index=True).sort_values("image_id", ignore_index=True)


def _restore_curated(paths: Paths) -> None:
    """data/curated as pinned. Only this flow writes it, under the labeling lock, so a difference
    means an earlier export died between writing files and `dvc add`. Its tasks are still open, so
    they are exported again from a clean start."""
    if not paths.has_curated:
        return
    if paths.curated.exists() and not is_clean(paths.root, paths.curated):
        logger.warning("data/curated differs from its pointer; restoring it")
        dvc_restore(paths.root, paths.curated)
    ensure_pinned(paths.root, [paths.curated])


def _current_rev(paths: Paths) -> str | None:
    """The rev as pinned now; None until the first label is exported."""
    return rev_of(pinned_md5s(paths.root, paths.versioned)) if paths.has_curated else None


@task(cache_policy=NO_CACHE)
def export_answers(
    paths: Paths, url: str, ls: LabelStudio, project: int, push: bool
) -> ExportResult:
    """Everything up to a new dataset rev. Safe to re-run after a crash at any point: open tasks
    are processed again, and identical content gives the identical rev."""
    _restore_curated(paths)
    answers = {
        int(t["id"]): answer
        for t in ls.list_tasks(project)
        if (answer := labeling.parse_answer(t.get("annotations", []), CLASSES)) is not None
    }
    tasks = db.read_tasks(url)
    fresh = [t for t in tasks if t.status == TASK_OPEN and t.ls_task_id in answers]
    changed = [
        t
        for t in tasks
        if t.status == TASK_EXPORTED
        and t.ls_task_id in answers
        and answers[t.ls_task_id] != t.labels
    ]
    if not fresh and not changed:
        return ExportResult(_current_rev(paths), 0, 0, None)

    ensure_pinned(paths.root, [paths.test_locked])
    locked = read_metadata(paths.test_locked)
    staging = paths.cache / "curated-staging"
    shutil.rmtree(staging, ignore_errors=True)
    (staging / "images").mkdir(parents=True)

    s3 = boto3.client("s3")
    rows: list[dict[str, Any]] = []
    gone: list[tuple[int, dict[str, int], str]] = []
    by_image: dict[str, db.TaskRow] = {}
    for t in fresh:
        image_id = labeling.image_id_for(t.sha256)
        target = staging / "images" / f"{image_id}{_EXTENSIONS[t.format]}"
        try:
            s3.download_file(INFERENCE_BUCKET, t.raw_key, str(target))
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code not in _NO_SUCH_OBJECT:
                raise  # an outage or a permission problem: nothing is written, tasks stay open
            already_exported = paths.curated / target.relative_to(staging)
            if not already_exported.exists():
                # Expired by the bucket's lifecycle rule: no file, nothing to train on.
                gone.append((t.ls_task_id, answers[t.ls_task_id], f"raw upload missing: {code}"))
                continue
            # An earlier export died after `dvc add`, so the row is in the data and needs its file.
            shutil.copyfile(already_exported, target)
        rows.append(curated_row(t, answers[t.ls_task_id], staging))
        by_image[image_id] = t

    accepted, refused = (
        partition_labels(as_metadata(rows), locked["patient_id"], staging)
        if rows
        else (as_metadata([]), {})
    )
    quarantine = gone + [
        (by_image[i].ls_task_id, answers[by_image[i].ls_task_id], why) for i, why in refused.items()
    ]
    if quarantine:
        logger.warning("quarantined %d labels: %s", len(quarantine), quarantine[:3])
        db.mark_quarantined(url, quarantine)
    updates = {labeling.image_id_for(t.sha256): answers[t.ls_task_id] for t in changed}
    if accepted.empty and not updates:
        return ExportResult(_current_rev(paths), 0, len(quarantine), None)

    existing = read_metadata(paths.curated) if paths.has_curated else None
    curated = apply_labels(existing, accepted, updates)
    (paths.curated / "images").mkdir(parents=True, exist_ok=True)
    for file in accepted["file"]:
        shutil.move(staging / file, paths.curated / file)
    write_metadata(curated, paths.curated)
    shutil.rmtree(staging, ignore_errors=True)

    dvc_add(paths.root, [paths.curated])
    if push:
        dvc_push(paths.root, [paths.curated])
    md5s = pinned_md5s(paths.root, paths.versioned)
    rev = rev_of(md5s)
    db.load_dataset_version(
        url,
        rev,
        md5s,
        {
            "data/nih": read_metadata(paths.nih),
            "data/test_locked": locked,
            "data/curated": curated,
        },
        git_commit(paths.root),
        str(runtime.flow_run.id or ""),
    )
    done = [
        (by_image[i].ls_task_id, answers[by_image[i].ls_task_id], i) for i in accepted["image_id"]
    ]
    done += [
        (t.ls_task_id, answers[t.ls_task_id], labeling.image_id_for(t.sha256)) for t in changed
    ]
    db.mark_exported(url, done, rev)
    logger.info(
        "dataset_rev %s: %d new labels, %d changed, %d quarantined; data/curated now holds %s",
        rev, len(accepted), len(changed), len(quarantine),
        curated["split"].value_counts().to_dict(),
    )  # fmt: skip
    logger.info("commit the pointer: git add data/curated.dvc && git commit")
    return ExportResult(rev, len(accepted), len(quarantine), None)


@task(cache_policy=NO_CACHE)
def decide_retrain(
    paths: Paths, url: str, rev: str, min_new_labels: int
) -> labeling.RetrainDecision:
    newest = newest_version(MlflowClient(require_mlflow_tracking_uri()), MODEL_NAME)
    baseline = "" if newest is None else str(newest.tags.get("dataset_rev", ""))
    curated = read_metadata(paths.curated)
    train_ids = set(curated.loc[curated["split"] == CURATED_TRAIN, "image_id"])
    return labeling.should_retrain(
        dataset_rev=rev,
        baseline_rev=baseline,
        last_triggered_rev=db.last_triggered_rev(url),
        new_labels=len(train_ids - db.curated_image_ids(url, baseline)),
        min_new_labels=min_new_labels,
    )


def start_retrain(cfg: LabelingConfig, rev: str, auto_promote: bool) -> str:
    """Queue the train deployment and return its flow run ID. `dataset_rev` pins the run: it fails
    rather than train on other data if the workspace has moved on by the time it starts.

    The idempotency key makes a rev's run unique in Prefect itself. If this process dies, or the
    decision can't be recorded, after Prefect created the run, the next pass gets that same run
    back and records it instead of starting another.
    """
    run = run_deployment(
        labeling.TRAIN_DEPLOYMENT,
        parameters={
            "config": cfg.train_config,
            "dataset_rev": rev,
            "request_promotion": True,
            "auto_promote": auto_promote,
        },
        timeout=0,
        as_subflow=False,
        idempotency_key=f"retrain-{rev}",
    )
    return str(run.id)  # type: ignore[union-attr]  # sync here: not awaited inside async code


def retrain_running(url: str, timeout_hours: float) -> bool:
    """Whether the newest retrain this flow started is still queued or running. One unfinished
    after `timeout_hours` is presumed dead: a killed worker leaves its run `Running` for good, and
    that mustn't stop exports forever."""
    run_id = db.last_train_run_id(url)
    if not run_id:
        return False
    with get_client(sync_client=True) as client:
        try:
            run = client.read_flow_run(UUID(run_id))
        except ObjectNotFound:
            return False
    if run.state is None or run.state.is_final() or run.created is None:
        return False
    return datetime.now(UTC) - run.created < timedelta(hours=timeout_hours)


def build_metrics(
    tasks: dict[str, int], curated: pd.DataFrame | None, decision: labeling.RetrainDecision | None
) -> CollectorRegistry:
    reg = CollectorRegistry()
    by_status = Gauge("cxr_labeling_tasks", "Labeling tasks by status", ["status"], registry=reg)
    for status in (TASK_OPEN, TASK_EXPORTED, TASK_QUARANTINED):
        by_status.labels(status).set(tasks.get(status, 0))
    images = Gauge("cxr_curated_images", "Images in data/curated by split", ["split"], registry=reg)
    for split in (CURATED_TRAIN, CURATED_RECENT):
        images.labels(split).set(0 if curated is None else int((curated["split"] == split).sum()))
    if decision is not None:
        Gauge("cxr_retrain_new_labels", "Curated training images the newest version doesn't have",
              registry=reg).set(decision.new_labels)  # fmt: skip
        Gauge("cxr_retrain_min_new_labels", "New labels a retrain needs (labeling.yaml)",
              registry=reg).set(decision.min_new_labels)  # fmt: skip
    return reg


@flow(name="label-export", log_prints=True)
def label_export_flow(
    config: str = "labeling",
    push: bool = True,
    trigger: bool = True,
    auto_promote: bool = False,
    min_new_labels: int | None = None,
    lock_held: bool = False,
) -> ExportResult:
    """`lock_held` means the caller already holds `db.labeling_lock`, as `make demo` does so that a
    scheduled export can't take its answers half-finished."""
    cfg = load_labeling_config(config)
    paths = Paths.discover()
    url = app_database_url()
    ls = LabelStudio.from_env()
    if ls is None:
        raise RuntimeError("LABEL_STUDIO_URL is not set; Label Studio is in the `full` profile")

    lock: AbstractContextManager[None] = nullcontext() if lock_held else db.labeling_lock(url)
    with lock:
        if retrain_running(url, cfg.retrain_timeout_hours):
            logger.info("should_retrain: no: the last retrain hasn't finished, so nothing exported")
            result = ExportResult(_current_rev(paths), 0, 0, None)
        else:
            project = ensure_review_project(ls, cfg.project_title)
            exported = export_answers(paths, url, ls, project, push)
            result = _decide_and_start(
                paths, url, cfg, exported, trigger, auto_promote, min_new_labels
            )

    curated = read_metadata(paths.curated) if paths.has_curated else None
    push_metrics(build_metrics(db.task_counts(url), curated, result.decision))
    return result


def _decide_and_start(
    paths: Paths,
    url: str,
    cfg: LabelingConfig,
    exported: ExportResult,
    trigger: bool,
    auto_promote: bool,
    min_new_labels: int | None,
) -> ExportResult:
    """`exported` with the retrain decision, after starting the retrain if it says yes."""
    if exported.dataset_rev is None:
        logger.info("should_retrain: no: no label has been exported yet, so there is no new rev")
        return exported
    decision = decide_retrain(
        paths, url, exported.dataset_rev, min_new_labels or cfg.min_new_labels
    )
    logger.info("should_retrain: %s: %s", "yes" if decision.trigger else "no", decision.reason)
    train_run, failure = "", None
    if decision.trigger and trigger:
        try:
            train_run = start_retrain(cfg, exported.dataset_rev, auto_promote)
        except Exception as exc:
            failure = exc
    _record_decision(url, decision, trigger, train_run, failure)
    if failure is not None:
        raise RetrainNotStarted(
            f"couldn't start {labeling.TRAIN_DEPLOYMENT}: {failure}"
        ) from failure
    return replace(exported, decision=decision, train_flow_run_id=train_run)


def _record_decision(
    url: str,
    decision: labeling.RetrainDecision,
    trigger: bool,
    train_run: str,
    failure: Exception | None,
) -> None:
    """A row whenever the outcome changes, so a scheduled pass that finds nothing new adds none.
    Whether it triggers or not, the flow run's log has the same line."""
    triggered = bool(train_run)
    reason = decision.reason
    if decision.trigger and not trigger:
        reason += "; not started (trigger is off)"
    elif failure is not None:
        reason += f"; couldn't start it: {failure}"
    last = db.latest_retrain_decision(url)
    same = last is not None and (
        last["dataset_rev"],
        last["new_labels"],
        bool(last["triggered"]),
        last["reason"],
    ) == (decision.dataset_rev, decision.new_labels, triggered, reason)
    if same:
        return
    db.record_retrain_decision(
        url,
        dataset_rev=decision.dataset_rev,
        baseline_rev=decision.baseline_rev,
        new_labels=decision.new_labels,
        min_new_labels=decision.min_new_labels,
        triggered=triggered,
        reason=reason,
        train_flow_run_id=train_run,
        prefect_flow_run_id=str(runtime.flow_run.id or ""),
    )


def push_metrics(registry: CollectorRegistry) -> None:
    url = os.environ.get("PUSHGATEWAY_URL")
    if url:
        push_to_gateway(url, job=PUSH_JOB, registry=registry)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--config", default="labeling", help="labeling config name or YAML path")
    parser.add_argument("--no-push", action="store_true", help="skip `dvc push`")
    parser.add_argument("--no-trigger", action="store_true", help="decide, but don't retrain")
    parser.add_argument(
        "--auto-promote", action="store_true", help="the retrain promotes itself if the gate passes"
    )
    parser.add_argument("--min-new-labels", type=int, help="override N from the config")
    args = parser.parse_args()
    result = label_export_flow(
        config=args.config,
        push=not args.no_push,
        trigger=not args.no_trigger,
        auto_promote=args.auto_promote,
        min_new_labels=args.min_new_labels,
    )
    print(
        json.dumps(
            {
                "dataset_rev": result.dataset_rev,
                "exported": result.exported,
                "quarantined": result.quarantined,
                "retrain": result.decision.trigger if result.decision else None,
                "reason": result.decision.reason if result.decision else None,
                "train_flow_run_id": result.train_flow_run_id,
            }
        )
    )


if __name__ == "__main__":
    main()
