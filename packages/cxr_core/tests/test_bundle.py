import json
from pathlib import Path

import numpy as np
import pytest

from cxr_core.bundle import MANIFEST_FILE, Manifest, load_head, save_head


def _manifest(**overrides: object) -> Manifest:
    fields: dict[str, object] = {
        "model_name": "cxr-classifier",
        "model_version": "3",
        "architecture": "resnet18",
        "classes": ("A", "B"),
        "image_size": 128,
        "preprocess_version": "1",
        "thresholds": {"A": 0.3, "B": 0.6},
        "lineage": {"dataset_rev": "abc"},
    }
    fields.update(overrides)
    return Manifest(**fields)  # type: ignore[arg-type]


def test_manifest_round_trip(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest.save(tmp_path)
    assert Manifest.load(tmp_path) == manifest


def test_thresholds_must_cover_exactly_the_classes() -> None:
    with pytest.raises(ValueError, match="thresholds"):
        _manifest(thresholds={"A": 0.3})


def test_unknown_bundle_format_is_rejected(tmp_path: Path) -> None:
    _manifest().save(tmp_path)
    data = json.loads((tmp_path / MANIFEST_FILE).read_text())
    data["bundle_format"] = 99
    (tmp_path / MANIFEST_FILE).write_text(json.dumps(data))
    with pytest.raises(ValueError, match="bundle_format"):
        Manifest.load(tmp_path)


def test_head_round_trip(tmp_path: Path) -> None:
    weight = np.arange(6, dtype=np.float64).reshape(2, 3)
    save_head(tmp_path, weight, np.array([0.5, -0.5]))
    w, b = load_head(tmp_path)
    assert w.dtype == np.float32 and b.dtype == np.float32
    np.testing.assert_array_equal(w, weight)
