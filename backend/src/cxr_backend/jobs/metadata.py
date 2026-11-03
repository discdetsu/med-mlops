"""Image metadata for metadata drift: patient sex, age, and view position.

The monitor flow compares these with the drift reference (the validation set's NIH metadata),
so they use the NIH vocabulary: sex `M` / `F`, age in years, view `PA` / `AP` (anything else is
kept as given, upper-cased, and shows up as a new category). Uploads may pass them as form
fields; for DICOM, missing fields come from the header. A PNG without fields has none, and the
monitor flow leaves it out of the metadata comparison.
"""

import logging
import re
from dataclasses import dataclass, replace
from typing import IO

import pydicom
from pydicom.errors import InvalidDicomError

logger = logging.getLogger(__name__)

MAX_AGE = 120
# DICOM AS: three digits and a unit, e.g. 045Y, 006M.
_DICOM_AGE = re.compile(r"^(\d{1,3})([DWMY])$")


@dataclass(frozen=True)
class ImageMetadata:
    patient_sex: str = ""
    patient_age: int | None = None
    view_position: str = ""

    def fill_from(self, other: ImageMetadata) -> ImageMetadata:
        """Fields missing here, taken from `other`."""
        return replace(
            self,
            patient_sex=self.patient_sex or other.patient_sex,
            patient_age=self.patient_age if self.patient_age is not None else other.patient_age,
            view_position=self.view_position or other.view_position,
        )


def normalize(
    patient_sex: str | None = None,
    patient_age: int | str | None = None,
    view_position: str | None = None,
) -> ImageMetadata:
    sex = (patient_sex or "").strip().upper()[:1]
    return ImageMetadata(
        patient_sex=sex if sex in ("M", "F") else "",
        patient_age=_age(patient_age),
        view_position=(view_position or "").strip().upper()[:16],
    )


def from_dicom(file: IO[bytes]) -> ImageMetadata:
    """Header tags only (no pixel data); empty metadata if the header can't be read. Leaves
    the file position where it was."""
    position = file.tell()
    try:
        ds = pydicom.dcmread(file, stop_before_pixels=True)
    except (InvalidDicomError, OSError, ValueError, EOFError) as exc:
        logger.info("no metadata from DICOM header: %s", exc)
        return ImageMetadata()
    finally:
        file.seek(position)
    return normalize(
        patient_sex=str(ds.get("PatientSex", "") or ""),
        patient_age=str(ds.get("PatientAge", "") or ""),
        view_position=str(ds.get("ViewPosition", "") or ""),
    )


def _age(value: int | str | None) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        text = value.strip().upper()
        match = _DICOM_AGE.match(text)
        if match:
            years = int(match.group(1)) if match.group(2) == "Y" else 0
            return years if years <= MAX_AGE else None
        if not text.isdigit():
            return None
        value = int(text)
    return value if 0 <= value <= MAX_AGE else None
