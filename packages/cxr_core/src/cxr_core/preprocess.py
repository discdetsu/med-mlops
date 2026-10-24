"""Raw PNG/DICOM → model input tensor. No torch: numpy for the math, pydicom and
Pillow only for decoding and resampling.

Bump PREPROCESS_VERSION on any change that alters the output array; it is stored
with every registered model and every prediction row.
"""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pydicom
from PIL import Image
from pydicom.pixels.processing import apply_modality_lut

PREPROCESS_VERSION = "1"

# The backbone is ImageNet-pretrained: grayscale is replicated to 3 channels and
# normalized with ImageNet statistics.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

FloatArray = npt.NDArray[np.float32]


def is_dicom(path: Path) -> bool:
    """DICOM Part 10 files carry `DICM` at byte 128, whatever their extension."""
    with path.open("rb") as f:
        header = f.read(132)
    return len(header) == 132 and header[128:] == b"DICM"


def load_image(path: Path) -> FloatArray:
    """Read a PNG or DICOM file into a 2D float32 array in [0, 1], bone bright."""
    return load_dicom(path) if is_dicom(path) else load_png(path)


def load_png(path: Path) -> FloatArray:
    with Image.open(path) as img:
        gray = img if img.mode in ("L", "I;16", "I") else img.convert("L")
        mode = gray.mode
        arr = np.asarray(gray)
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    if arr.dtype == np.uint16 or mode == "I;16":
        return arr.astype(np.float32) / 65535.0
    return min_max(arr.astype(np.float32))


def load_dicom(path: Path) -> FloatArray:
    """Modality LUT → linear VOI window (or min-max without one) → MONOCHROME1 inversion.

    Uncompressed transfer syntaxes only; compressed DICOM needs pylibjpeg or GDCM.
    """
    ds = pydicom.dcmread(path)
    photometric = ds.get("PhotometricInterpretation", "MONOCHROME2")
    if photometric not in ("MONOCHROME1", "MONOCHROME2"):
        raise ValueError(f"{path}: unsupported PhotometricInterpretation {photometric!r}")

    values = np.asarray(apply_modality_lut(ds.pixel_array, ds), dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"{path}: expected a single 2D frame, got shape {values.shape}")

    center, width = ds.get("WindowCenter"), ds.get("WindowWidth")
    if center is not None and width is not None:
        image = apply_window(values, _first(center), _first(width))
    else:
        image = min_max(values)

    # MONOCHROME1: the minimum value is displayed white, so invert to match MONOCHROME2.
    if photometric == "MONOCHROME1":
        image = 1.0 - image
    return image


def apply_window(values: FloatArray, center: float, width: float) -> FloatArray:
    """DICOM linear VOI window (PS3.3 C.11.2.1.2.1), mapped to [0, 1] instead of [ymin, ymax]."""
    if width < 1:
        raise ValueError(f"window width must be >= 1, got {width}")
    if width == 1:
        return (values > center - 0.5).astype(np.float32)
    out = (values - (center - 0.5)) / (width - 1) + 0.5
    windowed: FloatArray = np.clip(out, 0.0, 1.0).astype(np.float32)
    return windowed


def min_max(values: FloatArray) -> FloatArray:
    lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - lo) / (hi - lo)).astype(np.float32)


def resize(image: FloatArray, size: tuple[int, int]) -> FloatArray:
    """Bilinear resize of a 2D float array to (height, width), antialiased when shrinking."""
    height, width = size
    resized = Image.fromarray(np.ascontiguousarray(image, dtype=np.float32)).resize(
        (width, height), Image.Resampling.BILINEAR
    )
    return np.asarray(resized, dtype=np.float32)


def preprocess(image: FloatArray, size: int) -> FloatArray:
    """Turn a [0, 1] grayscale image into a (1, 3, size, size) normalized model input.

    The image is resized to a square without padding; NIH images are already square.
    """
    if image.ndim != 2:
        raise ValueError(f"expected a 2D grayscale image, got shape {image.shape}")
    if not np.isfinite(image).all():
        raise ValueError("image contains NaN or inf")

    resized = resize(image, (size, size))
    channels = (resized[None, :, :] - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
    return channels[None].astype(np.float32)


def _first(value: object) -> float:
    """WindowCenter/WindowWidth may be multi-valued; the first pair is the default."""
    if isinstance(value, pydicom.multival.MultiValue):
        return float(value[0])
    return float(value)  # type: ignore[arg-type]
