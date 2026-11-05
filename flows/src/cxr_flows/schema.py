"""Canonical metadata schema shared by every DVC dataset path, and the checks that guard it.

One row per image. The same columns appear in `data/nih/metadata.csv`,
`data/test_locked/metadata.csv`, and `data/curated/metadata.csv`, and are loaded into Postgres.
"""

from collections.abc import Collection, Mapping, Sequence
from pathlib import Path

import pandas as pd
import pandera.errors
import pandera.pandas as pa
import pydicom
from PIL import Image

from cxr_core.preprocess import is_dicom

# Chosen for enough positives in NIH ChestX-ray14 and for findings large enough to show
# at the demo's 128 px; see docs/model-card.md.
CLASSES = ("Atelectasis", "Cardiomegaly", "Effusion", "Mass", "Pneumothorax")

SPLITS = ("train", "val", "test_locked")
# data/curated (ADR 0009): `recent` is held back for evaluation and never trains.
CURATED_TRAIN = "train"
CURATED_RECENT = "recent"
CURATED_SPLITS = (CURATED_TRAIN, CURATED_RECENT)
LABEL_SOURCES = ("nih_nlp", "reviewer")
VIEW_POSITIONS = ("PA", "AP")
SEXES = ("M", "F")
MAX_AGE = 120  # NIH has a handful of ages > 400; ingest sets those to missing
MIN_IMAGE_SIDE = 256

COLUMNS = (
    "image_id",
    "file",  # path relative to the dataset directory, e.g. images/00000001_000.png
    "patient_id",
    "patient_sex",
    "patient_age",
    "view_position",
    "follow_up",
    "width",
    "height",
    "split",
    "label_source",
    *CLASSES,
)

_DTYPES: dict[str, str] = {
    "image_id": "str",
    "file": "str",
    "patient_id": "str",
    "patient_sex": "str",
    "patient_age": "Int64",
    "view_position": "str",
    "follow_up": "int64",
    "width": "int64",
    "height": "int64",
    "split": "str",
    "label_source": "str",
    **dict.fromkeys(CLASSES, "int64"),
}

METADATA_FILE = "metadata.csv"


class PatientLeakageError(ValueError):
    """A patient appears in more than one split or dataset path."""


def read_metadata(dataset_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(dataset_dir / METADATA_FILE, dtype=_DTYPES)
    return df.loc[:, list(COLUMNS)]


def write_metadata(df: pd.DataFrame, dataset_dir: Path) -> Path:
    path = dataset_dir / METADATA_FILE
    df.loc[:, list(COLUMNS)].sort_values("image_id").to_csv(path, index=False)
    return path


def metadata_schema(
    splits: Collection[str] = SPLITS,
    image_root: Path | None = None,
    extra_checks: Sequence[pa.Check] = (),
) -> pa.DataFrameSchema:
    """Schema for one dataset path. With `image_root`, every file must exist and match its
    recorded width and height (header read only, so it stays fast)."""
    checks = [
        pa.Check(
            lambda df: df.groupby("patient_id")["split"].transform("nunique") == 1,
            error="patient appears in more than one split",
        ),
        *extra_checks,
    ]
    if image_root is not None:
        checks.append(
            pa.Check(
                lambda df: pd.Series(
                    [
                        image_size(image_root / f) == (w, h)
                        for f, w, h in zip(df["file"], df["width"], df["height"], strict=True)
                    ],
                    index=df.index,
                ),
                error=f"image file missing, unreadable, or not the recorded size ({image_root})",
            )
        )

    binary = pa.Check.isin([0, 1])
    return pa.DataFrameSchema(
        {
            "image_id": pa.Column(str, pa.Check.str_length(min_value=1), unique=True),
            "file": pa.Column(str, pa.Check.str_length(min_value=1), unique=True),
            "patient_id": pa.Column(str, pa.Check.str_length(min_value=1)),
            "patient_sex": pa.Column(str, pa.Check.isin(SEXES)),
            "patient_age": pa.Column("Int64", pa.Check.in_range(0, MAX_AGE), nullable=True),
            "view_position": pa.Column(str, pa.Check.isin(VIEW_POSITIONS)),
            "follow_up": pa.Column(int, pa.Check.ge(0)),
            "width": pa.Column(int, pa.Check.ge(MIN_IMAGE_SIDE)),
            "height": pa.Column(int, pa.Check.ge(MIN_IMAGE_SIDE)),
            "split": pa.Column(str, pa.Check.isin(list(splits))),
            "label_source": pa.Column(str, pa.Check.isin(LABEL_SOURCES)),
            **{c: pa.Column(int, binary) for c in CLASSES},
        },
        checks=checks,
        strict=True,
        ordered=True,
    )


def check_no_shared_patients(frames: Mapping[str, pd.DataFrame]) -> None:
    """Raise if any patient appears in more than one of the named datasets."""
    owners: dict[str, list[str]] = {}
    for name, df in frames.items():
        for patient in df["patient_id"].unique():
            owners.setdefault(patient, []).append(name)
    shared = {p: names for p, names in owners.items() if len(names) > 1}
    if shared:
        sample = ", ".join(f"{p} in {names}" for p, names in list(shared.items())[:5])
        raise PatientLeakageError(f"{len(shared)} patients shared across datasets: {sample}")


def validate_dataset(
    df: pd.DataFrame, splits: Collection[str], image_root: Path | None = None
) -> pd.DataFrame:
    """Validate lazily so one run reports every failing check, not just the first."""
    return metadata_schema(splits, image_root).validate(df, lazy=True)


def validate_versioned(
    nih: pd.DataFrame, nih_dir: Path, locked: pd.DataFrame, locked_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate data/nih and data/test_locked, then check that no patient is in both."""
    nih = validate_dataset(nih, ("train", "val"), nih_dir)
    locked = validate_dataset(locked, ("test_locked",), locked_dir)
    check_no_shared_patients({"data/nih": nih, "data/test_locked": locked})
    return nih, locked


def validate_curated(
    curated: pd.DataFrame, curated_dir: Path, nih: pd.DataFrame, locked: pd.DataFrame
) -> pd.DataFrame:
    """Validate data/curated. Its patients may also be in data/nih's train split, but must never
    be in test_locked, and what it trains on must never be in val."""
    curated = validate_dataset(curated, CURATED_SPLITS, curated_dir)
    check_no_shared_patients({"data/curated": curated, "data/test_locked": locked})
    check_no_shared_patients(
        {
            "data/curated train": curated[curated["split"] == CURATED_TRAIN],
            "data/nih val": nih[nih["split"] == "val"],
        }
    )
    return curated


def partition_labels(
    df: pd.DataFrame, locked_patients: Collection[str], image_root: Path
) -> tuple[pd.DataFrame, dict[str, str]]:
    """New labels → the rows that may enter data/curated, and why each of the others may not.

    Any failed check quarantines only that row, by `image_id`, so one bad case never blocks the
    others. Without a row index in the failure (a broken column), the whole batch raises: that is
    a bug here, not bad data.
    """
    locked = frozenset(locked_patients)
    schema = metadata_schema(
        CURATED_SPLITS,
        image_root,
        [
            pa.Check(
                lambda frame: ~frame["patient_id"].isin(locked),
                error="patient is in test_locked",
            )
        ],
    )
    df = df.reset_index(drop=True)
    try:
        return schema.validate(df, lazy=True), {}
    except pandera.errors.SchemaErrors as errors:
        cases = errors.failure_cases
    rows = pd.to_numeric(cases["index"], errors="coerce")
    if rows.isna().any():
        raise ValueError(f"labels can't be checked row by row:\n{cases[rows.isna()]}")
    reasons: dict[str, set[str]] = {}
    for row, context, column, check in zip(
        rows.astype(int), cases["schema_context"], cases["column"], cases["check"], strict=True
    ):
        text = f"{column}: {check}" if context == "Column" else str(check)
        reasons.setdefault(str(df.at[row, "image_id"]), set()).add(text)
    keep = ~df["image_id"].isin(reasons)
    return df[keep].reset_index(drop=True), {k: "; ".join(sorted(v)) for k, v in reasons.items()}


def image_size(path: Path) -> tuple[int, int] | None:
    """(width, height) from the file's header, or None if it can't be read. Curated images are
    uploads as they arrived, so unlike the NIH sets they may be DICOM. Catches everything: pydicom
    raises many types on a malformed file, and one such file must not stop the other labels."""
    try:
        if is_dicom(path):
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            return int(ds.Columns), int(ds.Rows)
        with Image.open(path) as img:
            return img.size
    except Exception:
        return None
