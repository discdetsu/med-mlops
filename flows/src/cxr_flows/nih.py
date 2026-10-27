"""NIH ChestX-ray14 from Kaggle (`nih-chest-xrays/data`): metadata → canonical rows, patient
split, subset selection, and per-file image download (never the 42 GB of archives)."""

import hashlib
import logging
import os
import shutil
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from cxr_flows.schema import CLASSES, COLUMNS, MAX_AGE

KAGGLE_HANDLE = "nih-chest-xrays/data"
DATA_ENTRY_FILE = "Data_Entry_2017.csv"

# Kaggle's size for version 3 of the dataset, as one archive. Extracting needs the archive and
# the extracted files on disk at once.
ARCHIVE_BYTES = 45_077_186_807
_ARCHIVE_ROOT_MARKER = "nih-archive-root.txt"

# Kaggle keeps NIH's 12 archive folders. Data_Entry_2017.csv sorted by image name splits into
# the folders in this order and these sizes (checked against Kaggle at every boundary).
FOLDER_SIZES = (4999, *(10_000,) * 10, 7121)

# Split by a hash of the patient ID rather than a seeded shuffle: a patient's split never
# depends on which other patients are sampled, so it is stable across re-ingests, subset
# sizes, and new labels arriving later in Phase 5.
SPLIT_SALT = "cxr-split-v1"
SPLIT_CUTS = (("train", 70), ("val", 85), ("test_locked", 100))

log = logging.getLogger(__name__)


def patient_split(patient_id: str) -> str:
    bucket = int(hashlib.sha256(f"{SPLIT_SALT}:{patient_id}".encode()).hexdigest(), 16) % 100
    for split, cut in SPLIT_CUTS:
        if bucket < cut:
            return split
    raise AssertionError("unreachable: last cut is 100")


def read_data_entry(csv_path: Path) -> pd.DataFrame:
    """Data_Entry_2017.csv → canonical rows for all 112,120 images, plus `kaggle_path`.

    `width` and `height` are 0 until the file is downloaded and measured: the CSV's
    OriginalImage size is the pre-resize scan, not the 1024 px PNG.
    """
    raw = pd.read_csv(csv_path).sort_values("Image Index", ignore_index=True)
    if len(raw) != sum(FOLDER_SIZES):
        raise ValueError(f"{csv_path}: expected {sum(FOLDER_SIZES)} rows, got {len(raw)}")

    folders = np.repeat([f"images_{i:03d}" for i in range(1, 13)], FOLDER_SIZES)
    names = raw["Image Index"].astype(str)
    findings = raw["Finding Labels"].astype(str).str.split("|")
    patient_id = raw["Patient ID"].map(lambda p: f"{int(p):08d}")
    age = raw["Patient Age"]

    df = pd.DataFrame(
        {
            "image_id": names.str.removesuffix(".png"),
            "file": "images/" + names,
            "patient_id": patient_id,
            "patient_sex": raw["Patient Gender"].astype(str),
            "patient_age": age.where(age.between(0, MAX_AGE)).astype("Int64"),
            "view_position": raw["View Position"].astype(str),
            "follow_up": raw["Follow-up #"].astype("int64"),
            "width": 0,
            "height": 0,
            "split": patient_id.map(patient_split),
            "label_source": "nih_nlp",
            **{c: findings.map(lambda f, c=c: int(c in f)).astype("int64") for c in CLASSES},
        }
    )
    df["kaggle_path"] = pd.Series(folders) + "/images/" + names
    return df


def select_subset(
    entries: pd.DataFrame, positives_per_class: int | None, negatives: int | None, seed: int
) -> pd.DataFrame:
    """Up to `positives_per_class` images per class, plus `negatives` with no target class.

    None means no cap, so `None, None` selects every image. Images with other NIH findings
    (e.g. Infiltration only) count as negatives. Sampling always draws from the full table, so
    the same arguments give the same subset whether or not test_locked already exists.
    """
    rng = np.random.default_rng(seed)
    chosen: set[int] = set()
    for c in CLASSES:
        candidates = entries.index[(entries[c] == 1) & ~entries.index.isin(chosen)]
        chosen.update(_sample(rng, candidates, positives_per_class))

    is_negative = (entries[list(CLASSES)] == 0).all(axis=1)
    chosen.update(_sample(rng, entries.index[is_negative], negatives))
    return entries.loc[sorted(chosen)].reset_index(drop=True)


def _sample(rng: np.random.Generator, candidates: pd.Index, limit: int | None) -> list[int]:
    size = len(candidates) if limit is None else min(limit, len(candidates))
    picked: list[int] = rng.choice(candidates, size=size, replace=False).tolist()
    return picked


def parse_count(value: str) -> int | None:
    """argparse type for sample sizes: a positive integer, or `all` for no cap."""
    if value == "all":
        return None
    count = int(value)
    if count < 0:
        raise ValueError(f"expected a count >= 0 or 'all', got {value}")
    return count


def fetch_data_entry(staging: Path, archive: Path | None = None) -> Path:
    if archive is not None:
        return archive / DATA_ENTRY_FILE
    return _kaggle_file(DATA_ENTRY_FILE, staging)


def fetch_archive(cache_dir: Path) -> Path:
    """Download and extract the whole Kaggle dataset once (about 45 GB); return its root.

    Goes through kagglehub's own cache (KAGGLEHUB_CACHE=`cache_dir`) rather than `output_dir`:
    there the archive sits beside the extraction folder instead of inside it, so an interrupted
    download resumes where it stopped instead of starting over. kagglehub deletes the archive
    after extracting it.
    """
    marker = cache_dir / _ARCHIVE_ROOT_MARKER
    if marker.exists():
        root = Path(marker.read_text().strip())
        if (root / DATA_ENTRY_FILE).exists():
            return check_archive_root(root)

    cache_dir.mkdir(parents=True, exist_ok=True)
    partial = sum(p.stat().st_size for p in cache_dir.rglob("*.archive"))
    needed = 2 * ARCHIVE_BYTES - partial
    free = shutil.disk_usage(cache_dir).free
    if free < needed:
        raise RuntimeError(
            f"the full archive needs about {needed / 1e9:.0f} GB free in {cache_dir} while "
            f"extracting, but only {free / 1e9:.0f} GB is free; free some space, or download and "
            "unzip it elsewhere and pass that folder to --from-archive"
        )

    log.info(
        "downloading the full Kaggle archive (%.0f GB) into %s", ARCHIVE_BYTES / 1e9, cache_dir
    )
    os.environ["KAGGLEHUB_CACHE"] = str(cache_dir)  # read by kagglehub at call time
    import kagglehub

    root = check_archive_root(Path(kagglehub.dataset_download(KAGGLE_HANDLE)))
    marker.write_text(f"{root}\n")
    return root


def check_archive_root(root: Path) -> Path:
    """`root` must be an extracted copy of the Kaggle dataset: the CSV plus images_001..012."""
    missing = [
        name
        for name in (DATA_ENTRY_FILE, *(f"images_{i:03d}/images" for i in range(1, 13)))
        if not (root / name).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"{root} is not an extracted copy of {KAGGLE_HANDLE}; missing {', '.join(missing)}"
        )
    return root


def download_images(
    rows: pd.DataFrame,
    dataset_dir: Path,
    staging: Path,
    workers: int,
    archive: Path | None = None,
    link: bool = True,
) -> int:
    """Fetch each row's image to `dataset_dir/<file>`, skipping files already there.

    Without `archive`, files are downloaded one by one into `staging` (kagglehub's resumable
    per-file cache). With it, they come from the extracted archive and no network is used.
    Files from our own caches are hard-linked into the dataset when possible. Pass
    `link=False` for a folder someone else owns: `dvc add` moves dataset files into its cache
    and makes them read-only, which through a hard link would change their files too.
    Returns the number fetched.
    """
    todo = [
        (str(r.kaggle_path), dataset_dir / str(r.file))
        for r in rows.itertuples()
        if not (dataset_dir / str(r.file)).exists()
    ]
    if not todo:
        return 0
    (dataset_dir / "images").mkdir(parents=True, exist_ok=True)
    if archive is not None:
        log.info("linking %d images from the archive at %s", len(todo), archive)
    else:
        log.info("downloading %d images from Kaggle with %d workers", len(todo), workers)

    def fetch(item: tuple[str, Path]) -> None:
        source, dest = item
        src = archive / source if archive is not None else _kaggle_file(source, staging)
        _link_or_copy(src, dest, link)

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(fetch, item) for item in todo]):
            future.result()
            done += 1
            if done % (5000 if archive is not None else 250) == 0 or done == len(todo):
                log.info("fetched %d/%d", done, len(todo))
    return len(todo)


def measure_sizes(rows: pd.DataFrame, dataset_dir: Path) -> pd.DataFrame:
    """Fill `width` and `height` from the downloaded files' headers."""
    sizes = [_size(dataset_dir / str(f)) for f in rows["file"]]
    out = rows.copy()
    out["width"] = [w for w, _ in sizes]
    out["height"] = [h for _, h in sizes]
    return out


def canonical(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.loc[:, list(COLUMNS)].reset_index(drop=True)


def prune_images(dataset_dir: Path, keep: Iterable[str]) -> int:
    """Delete images not in `keep` (relative paths), so the DVC directory matches metadata."""
    wanted = set(keep)
    removed = 0
    for path in (dataset_dir / "images").glob("*.png"):
        if str(path.relative_to(dataset_dir)) not in wanted:
            path.unlink()
            removed += 1
    return removed


def _kaggle_file(path: str, staging: Path) -> Path:
    # Imported lazily: kagglehub reads credentials and config at import time.
    import kagglehub

    logging.getLogger("kagglehub").setLevel(logging.WARNING)
    return Path(kagglehub.dataset_download(KAGGLE_HANDLE, path=path, output_dir=str(staging)))


def _link_or_copy(src: Path, dest: Path, link: bool = True) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.unlink(missing_ok=True)
    try:
        if not link:
            raise OSError("copy requested")
        os.link(src, tmp)
    except OSError:
        shutil.copy2(src, tmp)
    tmp.replace(dest)


def _size(path: Path) -> tuple[int, int]:
    with Image.open(path) as img:
        return img.size
