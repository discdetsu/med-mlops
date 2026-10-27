"""Canonical metadata schema shared by every DVC dataset path, and the checks that guard it.

One row per image. The same columns appear in `data/nih/metadata.csv`,
`data/test_locked/metadata.csv`, and later `data/curated/`, and are loaded into Postgres.
"""

from collections.abc import Collection, Mapping
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from PIL import Image

# Chosen for enough positives in NIH ChestX-ray14 and for findings large enough to show
# at the demo's 128 px; see docs/model-card.md.
CLASSES = ("Atelectasis", "Cardiomegaly", "Effusion", "Mass", "Pneumothorax")

SPLITS = ("train", "val", "test_locked")
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
    splits: Collection[str] = SPLITS, image_root: Path | None = None
) -> pa.DataFrameSchema:
    """Schema for one dataset path. With `image_root`, every file must exist and match its
    recorded width and height (header read only, so it stays fast)."""
    checks = [
        pa.Check(
            lambda df: df.groupby("patient_id")["split"].transform("nunique") == 1,
            error="patient appears in more than one split",
        )
    ]
    if image_root is not None:
        checks.append(
            pa.Check(
                lambda df: pd.Series(
                    [
                        _image_size(image_root / f) == (w, h)
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


def _image_size(path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as img:
            return img.size
    except OSError:
        return None
