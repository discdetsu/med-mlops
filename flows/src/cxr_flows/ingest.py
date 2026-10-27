"""Prefect `ingest` flow: NIH subset → validate → `dvc add` + `dvc push` → Postgres.

    uv run python -m cxr_flows.ingest [--positives-per-class N|all] [--negatives N|all]
                                      [--from-archive [DIR]]

By default only the sampled images are downloaded, one file at a time. `--from-archive`
downloads Kaggle's whole archive once instead (about 45 GB, roughly 90 GB free needed while
extracting; resumable) into .cache/cxr/kagglehub, or uses an already extracted copy at DIR.
That is the fast path for large subsets; `--positives-per-class all --negatives all` takes
every image.

`data/test_locked` and `data/golden` are created on the first run only. After that they are
frozen: later runs pull and check them but never rewrite them, and sampled images from
test_locked patients are dropped instead of added.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from prefect import flow, get_run_logger, runtime, task
from prefect.cache_policies import NO_CACHE

from cxr_flows import golden, nih
from cxr_flows.db import load_dataset_version
from cxr_flows.schema import (
    check_no_shared_patients,
    read_metadata,
    validate_dataset,
    write_metadata,
)
from cxr_flows.settings import Paths, app_database_url
from cxr_flows.versioning import (
    DatasetChangedError,
    dataset_rev,
    dvc_add,
    dvc_file,
    dvc_md5,
    dvc_pull,
    dvc_push,
    git_commit,
    is_clean,
)


@dataclass(frozen=True)
class Datasets:
    nih: pd.DataFrame
    test_locked: pd.DataFrame
    golden: pd.DataFrame


@task(cache_policy=NO_CACHE)
def ensure_frozen(paths: Paths) -> bool:
    """Pull test_locked and golden if they exist in DVC; fail if the workspace copy differs."""
    if not dvc_file(paths.test_locked).exists():
        return False
    for target in (paths.test_locked, paths.golden):
        if not target.exists():
            dvc_pull(paths.root, [target])
        if not is_clean(paths.root, target):
            rel = target.relative_to(paths.root)
            raise DatasetChangedError(
                f"{rel} differs from {rel}.dvc but is frozen; restore it with `dvc checkout {rel}`"
            )
    return True


@task(cache_policy=NO_CACHE)
def locate_archive(paths: Paths, archive_dir: str | None) -> Path:
    """An extracted copy of the Kaggle dataset: `archive_dir`, or downloaded into the cache."""
    if archive_dir:
        return nih.check_archive_root(Path(archive_dir).expanduser().resolve())
    return nih.fetch_archive(paths.cache / "kagglehub")


@task(cache_policy=NO_CACHE)
def select(
    paths: Paths,
    positives_per_class: int | None,
    negatives: int | None,
    seed: int,
    archive: Path | None,
) -> pd.DataFrame:
    entries = nih.read_data_entry(nih.fetch_data_entry(paths.cache / "kaggle", archive))
    subset = nih.select_subset(entries, positives_per_class, negatives, seed)
    get_run_logger().info(
        "selected %d images from %d patients: %s",
        len(subset),
        subset["patient_id"].nunique(),
        subset["split"].value_counts().to_dict(),
    )
    return subset


@task(cache_policy=NO_CACHE)
def materialize(
    paths: Paths,
    subset: pd.DataFrame,
    frozen: bool,
    workers: int,
    archive: Path | None,
    link: bool,
) -> Datasets:
    """Fetch images into the DVC directories and write their metadata."""
    staging = paths.cache / "kaggle"
    logger = get_run_logger()

    train_val = subset[subset["split"].isin(["train", "val"])]
    nih.download_images(train_val, paths.nih, staging, workers, archive, link)
    if removed := nih.prune_images(paths.nih, train_val["file"]):
        logger.info("removed %d images no longer in the subset from data/nih", removed)
    nih_df = nih.canonical(nih.measure_sizes(train_val, paths.nih))
    write_metadata(nih_df, paths.nih)

    if frozen:
        logger.info("data/test_locked is frozen; keeping it as is")
    else:
        locked = subset[subset["split"] == "test_locked"]
        nih.download_images(locked, paths.test_locked, staging, workers, archive, link)
        locked_df = nih.canonical(nih.measure_sizes(locked, paths.test_locked))
        write_metadata(locked_df, paths.test_locked)
        golden.build_golden(golden.pick_golden(locked_df), paths.test_locked, paths.golden)
        logger.info("created data/test_locked (%d images) and data/golden", len(locked_df))

    return Datasets(
        nih=nih_df,
        test_locked=read_metadata(paths.test_locked),
        golden=golden.read_manifest(paths.golden),
    )


@task(cache_policy=NO_CACHE)
def validate(paths: Paths, data: Datasets) -> None:
    validate_dataset(data.nih, ("train", "val"), paths.nih)
    validate_dataset(data.test_locked, ("test_locked",), paths.test_locked)
    check_no_shared_patients({"data/nih": data.nih, "data/test_locked": data.test_locked})
    missing = set(data.golden["image_id"]) - set(data.test_locked["image_id"])
    if missing:
        raise ValueError(f"golden images not from test_locked: {sorted(missing)}")


@task(cache_policy=NO_CACHE)
def version(paths: Paths, frozen: bool, push: bool) -> dict[str, str]:
    """`dvc add` (+ `dvc push`) and return the pinned md5 of each dataset path."""
    targets = [paths.nih] if frozen else [paths.nih, paths.test_locked, paths.golden]
    dvc_add(paths.root, targets)
    if push:
        dvc_push(paths.root, [paths.nih, paths.test_locked, paths.golden])
    return {str(t.relative_to(paths.root)): dvc_md5(t) for t in (paths.nih, paths.test_locked)}


@flow(name="ingest", log_prints=True)
def ingest_flow(
    positives_per_class: int | None = 500,
    negatives: int | None = 1500,
    seed: int = 0,
    workers: int = 8,
    push: bool = True,
    from_archive: bool = False,
    archive_dir: str | None = None,
) -> str:
    """Returns the new dataset_rev. None for a count means every matching image."""
    paths = Paths.discover()
    logger = get_run_logger()

    frozen = ensure_frozen(paths)
    archive = locate_archive(paths, archive_dir) if from_archive or archive_dir else None
    subset = select(paths, positives_per_class, negatives, seed, archive)
    # Hard-link only from our own caches, never from a folder the user pointed us at.
    data = materialize(paths, subset, frozen, workers, archive, link=not archive_dir)
    validate(paths, data)
    md5s = version(paths, frozen, push)

    rev = dataset_rev(md5s)
    load_dataset_version(
        app_database_url(),
        rev,
        md5s,
        {"data/nih": data.nih, "data/test_locked": data.test_locked},
        git_commit(paths.root),
        runtime.flow_run.id,
    )
    logger.info("dataset_rev %s: %s", rev, md5s)
    logger.info("commit the pointers: git add data/*.dvc data/.gitignore && git commit")
    return rev


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--positives-per-class", type=nih.parse_count, default=500, metavar="N|all")
    parser.add_argument("--negatives", type=nih.parse_count, default=1500, metavar="N|all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-push", action="store_true", help="skip `dvc push`")
    parser.add_argument(
        "--from-archive",
        nargs="?",
        const="",
        metavar="DIR",
        help="use Kaggle's full archive: download it (no DIR) or read an extracted copy at DIR",
    )
    args = parser.parse_args()
    ingest_flow(
        positives_per_class=args.positives_per_class,
        negatives=args.negatives,
        seed=args.seed,
        workers=args.workers,
        push=not args.no_push,
        from_archive=args.from_archive is not None,
        archive_dir=args.from_archive or None,
    )


if __name__ == "__main__":
    main()
