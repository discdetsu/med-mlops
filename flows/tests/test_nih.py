from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cxr_flows.nih import FOLDER_SIZES, patient_split, read_data_entry, select_subset
from cxr_flows.schema import CLASSES

N = sum(FOLDER_SIZES)


@pytest.fixture(scope="module")
def data_entry(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A synthetic Data_Entry_2017.csv with the real shape and column names, shuffled."""
    rng = np.random.default_rng(0)
    findings = np.array(
        ["No Finding", "Cardiomegaly", "Effusion|Mass", "Infiltration", "Atelectasis|Pneumothorax"]
    )
    patient = np.arange(N) // 4 + 1
    df = pd.DataFrame(
        {
            "Image Index": [f"{p:08d}_{i % 4:03d}.png" for i, p in enumerate(patient)],
            "Finding Labels": findings[rng.integers(0, len(findings), N)],
            "Follow-up #": np.arange(N) % 4,
            "Patient ID": patient,
            "Patient Age": rng.integers(1, 95, N),
            "Patient Gender": rng.choice(["M", "F"], N),
            "View Position": rng.choice(["PA", "AP"], N),
            "OriginalImage[Width": 2500,
            "Height]": 2048,
            "OriginalImagePixelSpacing[x": 0.14,
            "y]": 0.14,
            "Unnamed: 11": np.nan,
        }
    )
    df.loc[7, "Patient Age"] = 414  # NIH has a few of these
    path = tmp_path_factory.mktemp("nih") / "Data_Entry_2017.csv"
    df.sample(frac=1, random_state=0).to_csv(path, index=False)
    return path


def test_patient_split_is_stable_and_roughly_70_15_15() -> None:
    assert patient_split("00000042") == patient_split("00000042")
    counts = Counter(patient_split(f"{i:08d}") for i in range(20_000))
    assert counts["train"] / 20_000 == pytest.approx(0.70, abs=0.02)
    assert counts["val"] / 20_000 == pytest.approx(0.15, abs=0.02)
    assert counts["test_locked"] / 20_000 == pytest.approx(0.15, abs=0.02)


def test_read_data_entry_maps_folders_at_batch_boundaries(data_entry: Path) -> None:
    df = read_data_entry(data_entry)
    assert len(df) == N
    assert df["image_id"].is_monotonic_increasing
    folder = df["kaggle_path"].str.split("/").str[0]
    assert folder.iloc[0] == "images_001"
    assert folder.iloc[4998] == "images_001"
    assert folder.iloc[4999] == "images_002"
    assert folder.iloc[14999] == "images_003"
    assert folder.iloc[-1] == "images_012"
    assert df["kaggle_path"].iloc[0] == "images_001/images/00000001_000.png"


def test_read_data_entry_canonical_values(data_entry: Path) -> None:
    df = read_data_entry(data_entry)
    assert df.loc[7, "patient_age"] is pd.NA
    assert df["patient_id"].iloc[0] == "00000001"
    assert df["file"].iloc[0] == "images/00000001_000.png"
    assert set(df["label_source"]) == {"nih_nlp"}
    # A patient's images all land in the same split.
    assert (df.groupby("patient_id")["split"].nunique() == 1).all()
    infiltration_only = df[(df[list(CLASSES)] == 0).all(axis=1)]
    assert len(infiltration_only) > 0  # other findings count as negatives


def test_rejects_unexpected_row_count(tmp_path: Path, data_entry: Path) -> None:
    short = tmp_path / "short.csv"
    pd.read_csv(data_entry).iloc[:-1].to_csv(short, index=False)
    with pytest.raises(ValueError, match="expected"):
        read_data_entry(short)


def test_select_subset_quotas_and_determinism(data_entry: Path) -> None:
    entries = read_data_entry(data_entry)
    a = select_subset(entries, positives_per_class=50, negatives=80, seed=0)
    b = select_subset(entries, positives_per_class=50, negatives=80, seed=0)
    pd.testing.assert_frame_equal(a, b)
    assert a["image_id"].is_unique
    for c in CLASSES:
        assert a[c].sum() >= 50
    assert (a[list(CLASSES)] == 0).all(axis=1).sum() == 80
    c = select_subset(entries, positives_per_class=50, negatives=80, seed=1)
    assert set(a["image_id"]) != set(c["image_id"])


def test_select_subset_without_caps_takes_everything(data_entry: Path) -> None:
    entries = read_data_entry(data_entry)
    everything = select_subset(entries, positives_per_class=None, negatives=None, seed=0)
    assert len(everything) == len(entries)
