"""E2E parity in CI: a small random model on synthetic golden files (no NIH data needed).

The train flow runs the same check on the real golden set before registering a model.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from cxr_core.model import CamModel, build_model
from cxr_core.preprocess import load_image
from cxr_flows.config import ParityTolerance
from cxr_flows.export import export_onnx, run_parity
from cxr_flows.golden import write_dicom

SIZE = 64


@pytest.fixture(scope="module")
def model() -> CamModel:
    torch.manual_seed(0)
    return build_model("resnet18", num_classes=5).eval()


@pytest.fixture
def golden(tmp_path: Path, phantom: np.ndarray) -> list[tuple[Path, str]]:
    files = []
    for image_id, pixels in (("a", phantom), ("b", np.flipud(phantom).copy())):
        png = tmp_path / f"{image_id}.png"
        Image.fromarray(pixels).save(png)
        files.append((png, image_id))
        for photometric in ("MONOCHROME2", "MONOCHROME1"):
            dcm = write_dicom(tmp_path / f"{image_id}_{photometric}.dcm", pixels, photometric)
            files.append((dcm, image_id))
    return files


def test_golden_dicom_decodes_like_the_png(golden: list[tuple[Path, str]]) -> None:
    png = load_image(golden[0][0])
    for path, _ in golden[1:3]:
        np.testing.assert_allclose(load_image(path), png, atol=1e-6)
    # MONOCHROME1 really is stored inverted.
    assert "MONOCHROME1" in golden[2][0].name


def test_parity_passes_for_a_faithful_export(
    model: CamModel, golden: list[tuple[Path, str]], tmp_path: Path
) -> None:
    onnx_path = export_onnx(model, SIZE, tmp_path / "model.onnx")
    report = run_parity(model, onnx_path, golden, SIZE, ParityTolerance())
    assert report.passed, report.failures
    assert len(report.files) == 6
    assert set(report.cross_format) == {"a", "b"}


def test_parity_catches_a_model_that_drifted_from_its_export(
    golden: list[tuple[Path, str]], tmp_path: Path
) -> None:
    torch.manual_seed(1)
    model = build_model("resnet18", num_classes=5).eval()
    onnx_path = export_onnx(model, SIZE, tmp_path / "model.onnx")
    with torch.no_grad():
        model.head.bias += 0.5
    report = run_parity(model, onnx_path, golden, SIZE, ParityTolerance())
    assert not report.passed
    assert any("probabilities" in f for f in report.failures)
