"""Golden set for the E2E parity test: a few real test_locked images, each stored as PNG, as
DICOM MONOCHROME2, and as DICOM MONOCHROME1. All three decode to the same [0, 1] array
through `cxr_core.preprocess`, so the parity test also checks MONOCHROME1 inversion and
windowing end to end.
"""

import shutil
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from PIL import Image
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import UID, ExplicitVRLittleEndian, generate_uid

from cxr_flows.schema import CLASSES

MANIFEST_FILE = "manifest.csv"
# One positive for each of these, plus one image with no target finding.
GOLDEN_POSITIVES = ("Cardiomegaly", "Effusion")
DIGITAL_XRAY_FOR_PRESENTATION = UID("1.2.840.10008.5.1.4.1.1.1.1")

# 8-bit PNG values x are stored as 12-bit 16x. With center 2040.5 and width 4081, the DICOM
# linear window maps stored value v to v / 4080, which is exactly x / 255 again.
_SCALE = 16
_MAX_STORED = 255 * _SCALE
_WINDOW_CENTER = _MAX_STORED / 2 + 0.5
_WINDOW_WIDTH = _MAX_STORED + 1


def pick_golden(locked: pd.DataFrame) -> pd.DataFrame:
    """Deterministic picks: the first image (by id) for each positive, then one negative."""
    ordered = locked.sort_values("image_id")
    picks: list[str] = []
    for c in GOLDEN_POSITIVES:
        picks.append(
            ordered.loc[(ordered[c] == 1) & ~ordered["image_id"].isin(picks)].iloc[0]["image_id"]
        )
    negative = (ordered[list(CLASSES)] == 0).all(axis=1)
    picks.append(ordered.loc[negative].iloc[0]["image_id"])
    return ordered.set_index("image_id").loc[picks].reset_index()


def build_golden(picks: pd.DataFrame, source_dir: Path, golden_dir: Path) -> pd.DataFrame:
    """Write PNG + DICOM (MONOCHROME2 and MONOCHROME1) for each pick and a manifest."""
    (golden_dir / "png").mkdir(parents=True, exist_ok=True)
    (golden_dir / "dicom").mkdir(parents=True, exist_ok=True)
    rows = []
    for pick in picks.itertuples():
        image_id = str(pick.image_id)
        png = golden_dir / "png" / f"{image_id}.png"
        shutil.copyfile(source_dir / str(pick.file), png)
        rows.append((f"png/{png.name}", image_id, "png", ""))

        with Image.open(png) as img:
            pixels = np.asarray(img.convert("L"))
        for photometric in ("MONOCHROME2", "MONOCHROME1"):
            name = f"dicom/{image_id}_{photometric.lower()}.dcm"
            write_dicom(
                golden_dir / name,
                pixels,
                photometric,
                patient_id=str(pick.patient_id),
                sex=str(pick.patient_sex),
                view=str(pick.view_position),
            )
            rows.append((name, image_id, "dicom", photometric))

    manifest = pd.DataFrame(rows, columns=["file", "image_id", "format", "photometric"])
    manifest.to_csv(golden_dir / MANIFEST_FILE, index=False)
    return manifest


def read_manifest(golden_dir: Path) -> pd.DataFrame:
    return pd.read_csv(golden_dir / MANIFEST_FILE, dtype=str, keep_default_na=False)


def write_dicom(
    path: Path,
    pixels: npt.NDArray[np.uint8],
    photometric: str,
    *,
    patient_id: str = "golden",
    sex: str = "O",
    view: str = "PA",
) -> Path:
    """Single-frame uncompressed 12-bit DX image whose window maps back to `pixels / 255`."""
    if pixels.dtype != np.uint8 or pixels.ndim != 2:
        raise ValueError(f"expected 2D uint8 pixels, got {pixels.dtype} {pixels.shape}")
    stored = pixels.astype(np.uint16) * _SCALE
    if photometric == "MONOCHROME1":
        stored = _MAX_STORED - stored  # white is the minimum value in MONOCHROME1
    elif photometric != "MONOCHROME2":
        raise ValueError(f"unsupported photometric interpretation {photometric!r}")

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = DIGITAL_XRAY_FOR_PRESENTATION
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = "DX"
    ds.PatientID = patient_id
    ds.PatientSex = sex
    ds.ViewPosition = view
    ds.Rows, ds.Columns = stored.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photometric
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 0
    ds.WindowCenter = _WINDOW_CENTER
    ds.WindowWidth = _WINDOW_WIDTH
    ds.PixelData = stored.astype("<u2").tobytes()
    ds.save_as(path, enforce_file_format=True)
    return path
