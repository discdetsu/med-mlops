from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from cxr_flows.schema import CLASSES, COLUMNS


def make_rows(n: int, split: str = "train", patient_offset: int = 0) -> pd.DataFrame:
    """`n` canonical rows, one patient per two images, all labels negative."""
    rows = pd.DataFrame(
        {
            "image_id": [f"{patient_offset + i // 2:08d}_{i % 2:03d}" for i in range(n)],
            "file": [f"images/{patient_offset + i // 2:08d}_{i % 2:03d}.png" for i in range(n)],
            "patient_id": [f"{patient_offset + i // 2:08d}" for i in range(n)],
            "patient_sex": ["M", "F"] * (n // 2) + ["M"] * (n % 2),
            "patient_age": pd.array([40 + i for i in range(n)], dtype="Int64"),
            "view_position": "PA",
            "follow_up": [i % 2 for i in range(n)],
            "width": 256,
            "height": 256,
            "split": split,
            "label_source": "nih_nlp",
            **dict.fromkeys(CLASSES, 0),
        }
    )
    return rows.loc[:, list(COLUMNS)]


def write_images(rows: pd.DataFrame, root: Path) -> None:
    (root / "images").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for f, w, h in zip(rows["file"], rows["width"], rows["height"], strict=True):
        Image.fromarray(rng.integers(0, 256, (h, w), dtype=np.uint8)).save(root / f)


RowsFactory = Callable[..., pd.DataFrame]
ImageWriter = Callable[[pd.DataFrame, Path], None]


@pytest.fixture
def rows() -> RowsFactory:
    return make_rows


@pytest.fixture
def images() -> ImageWriter:
    return write_images


@pytest.fixture
def phantom() -> np.ndarray:
    """A smooth 8-bit chest-like phantom: bright ellipse body, two darker lungs."""
    yy, xx = np.mgrid[0:256, 0:256] / 255.0
    body = ((xx - 0.5) / 0.45) ** 2 + ((yy - 0.5) / 0.48) ** 2 < 1
    lung_l = ((xx - 0.33) / 0.13) ** 2 + ((yy - 0.45) / 0.3) ** 2 < 1
    lung_r = ((xx - 0.67) / 0.13) ** 2 + ((yy - 0.45) / 0.3) ** 2 < 1
    img = 30 + 170 * body - 120 * (lung_l | lung_r) + 40 * yy
    return np.clip(img, 0, 255).astype(np.uint8)
