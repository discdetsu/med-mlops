from pathlib import Path
from typing import Any

import pandas as pd
import pandera.errors
import pytest

from cxr_flows.schema import (
    PatientLeakageError,
    check_no_shared_patients,
    image_size,
    read_metadata,
    validate_curated,
    validate_dataset,
    write_metadata,
)


def test_valid_frame_passes(tmp_path: Path, rows: Any, images: Any) -> None:
    df = rows(6)
    images(df, tmp_path)
    validate_dataset(df, ("train", "val"), tmp_path)


def test_metadata_csv_round_trip_keeps_types(tmp_path: Path, rows: Any) -> None:
    df = rows(4)
    df.loc[1, "patient_age"] = pd.NA
    write_metadata(df, tmp_path)
    back = read_metadata(tmp_path)
    assert back["patient_id"].iloc[0] == "00000000"  # leading zeros survive
    assert back["patient_age"].isna().sum() == 1
    validate_dataset(back, ("train",))


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("view_position", "LL"),
        ("patient_sex", "X"),
        ("patient_age", 414),
        ("Effusion", 2),
        ("label_source", "guess"),
        ("split", "test_locked"),
        ("width", 64),
    ],
)
def test_disallowed_values_fail(column: str, value: object, rows: Any) -> None:
    df = rows(4)
    df.loc[0, column] = value
    with pytest.raises(pandera.errors.SchemaErrors):
        validate_dataset(df, ("train", "val"))


def test_patient_in_two_splits_fails(rows: Any) -> None:
    df = rows(4)
    df.loc[1, "split"] = "val"  # rows 0 and 1 share a patient
    with pytest.raises(pandera.errors.SchemaErrors, match="more than one split"):
        validate_dataset(df, ("train", "val"))


def test_missing_or_resized_image_fails(tmp_path: Path, rows: Any, images: Any) -> None:
    df = rows(4)
    images(df, tmp_path)
    (tmp_path / df["file"].iloc[0]).unlink()
    df.loc[1, "width"] = 512
    with pytest.raises(pandera.errors.SchemaErrors) as info:
        validate_dataset(df, ("train",), tmp_path)
    failed = info.value.failure_cases["index"].tolist()
    assert 0 in failed and 1 in failed


def test_extra_column_fails(rows: Any) -> None:
    with pytest.raises(pandera.errors.SchemaErrors):
        validate_dataset(rows(2).assign(extra=1), ("train",))


def test_patient_shared_with_test_locked_raises(rows: Any) -> None:
    train = rows(4)
    locked = rows(2, split="test_locked", patient_offset=1)  # patient 1 is in both
    with pytest.raises(PatientLeakageError, match="00000001"):
        check_no_shared_patients({"data/nih": train, "data/test_locked": locked})


def test_disjoint_patients_pass(rows: Any) -> None:
    check_no_shared_patients({"data/nih": rows(4), "data/test_locked": rows(4, "test_locked", 100)})


def test_image_size_reads_png_and_dicom_headers(tmp_path: Path) -> None:
    golden = Path(__file__).parents[2] / "data/golden"
    png, dcm = golden / "png/00000001_000.png", golden / "dicom/00000001_000_monochrome1.dcm"
    if not (png.exists() and dcm.exists()):
        pytest.skip("golden set not pulled")
    assert image_size(png) == image_size(dcm) == (1024, 1024)


def test_image_size_of_a_bad_file_is_none(tmp_path: Path) -> None:
    (tmp_path / "bad.png").write_bytes(b"not an image")
    assert image_size(tmp_path / "bad.png") is None
    assert image_size(tmp_path / "missing.png") is None
    (tmp_path / "bad.dcm").write_bytes(b"\0" * 128 + b"DICM")
    assert image_size(tmp_path / "bad.dcm") is None


def test_curated_rows_may_share_patients_with_train_but_not_val_or_locked(rows: Any) -> None:
    nih = rows(4)
    nih.loc[2:, "split"] = "val"  # patient 0 in train, patient 1 in val
    locked = rows(2, "test_locked", 500)

    def curated(patient: str, split: str = "train") -> Any:
        df = rows(1).assign(label_source="reviewer", split=split, patient_id=patient)
        return df

    validate_curated(curated("00000000"), None, nih, locked)  # type: ignore[arg-type]
    validate_curated(curated("00000001", "recent"), None, nih, locked)  # type: ignore[arg-type]
    with pytest.raises(PatientLeakageError, match="val"):
        validate_curated(curated("00000001"), None, nih, locked)  # type: ignore[arg-type]
    with pytest.raises(PatientLeakageError, match="test_locked"):
        validate_curated(curated("00000500"), None, nih, locked)  # type: ignore[arg-type]
