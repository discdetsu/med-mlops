from pathlib import Path
from typing import Any

import pandas as pd
import pandera.errors
import pytest

from cxr_flows.schema import (
    PatientLeakageError,
    check_no_shared_patients,
    read_metadata,
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
