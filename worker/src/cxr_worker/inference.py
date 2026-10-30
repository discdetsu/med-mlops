"""One image through the model: the same `cxr_core` code path the E2E parity test checks."""

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from cxr_core.cam import class_activation_map
from cxr_core.inference import FloatArray, run_onnx, sigmoid
from cxr_core.preprocess import load_image, preprocess, resize
from cxr_worker.model import LoadedModel


class BadImage(Exception):
    """The upload can't be decoded or preprocessed. Retrying won't help."""


@dataclass(frozen=True)
class Inference:
    probabilities: dict[str, float]
    display: FloatArray  # [0, 1], (h, w): the decoded image, windowed and downscaled
    heatmaps: dict[str, FloatArray]  # [0, 1], same (h, w) as display


def run_inference(model: LoadedModel, path: Path, display_size: int) -> Inference:
    manifest = model.manifest
    try:
        image = load_image(path)
        x = preprocess(image, manifest.image_size)
    except Exception as exc:  # pydicom, Pillow, and our own checks all raise different types
        # Reported back to the user, who never saw the worker's temp path.
        message = str(exc).replace(str(path), "the upload")
        raise BadImage(f"{type(exc).__name__}: {message}") from exc

    logits, features = run_onnx(model.session, x)
    probs = sigmoid(logits)[0]
    size = display_shape(image.shape, display_size)
    return Inference(
        probabilities={c: float(probs[k]) for k, c in enumerate(manifest.classes)},
        display=resize(image, size),
        heatmaps={
            c: class_activation_map(features[0], model.head_weight, k, size)
            for k, c in enumerate(manifest.classes)
        },
    )


def display_shape(shape: tuple[int, ...], long_side: int) -> tuple[int, int]:
    """(height, width) with the long side at most `long_side`, aspect ratio kept."""
    height, width = shape[0], shape[1]
    scale = min(1.0, long_side / max(height, width))
    return max(1, round(height * scale)), max(1, round(width * scale))


def to_png(image: FloatArray) -> bytes:
    """[0, 1] float → 8-bit grayscale PNG."""
    pixels = (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, format="PNG", optimize=True)
    return buf.getvalue()
