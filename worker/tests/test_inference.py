from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cxr_core.preprocess import load_image, preprocess
from cxr_worker.inference import BadImage, display_shape, run_inference, to_png
from cxr_worker.model import load_bundle

from .conftest import CLASSES, IMAGE_SIZE, BundleFactory


def test_probabilities_follow_the_gap_linear_head(
    make_bundle: BundleFactory, golden_png: Path
) -> None:
    model = load_bundle(make_bundle())
    result = run_inference(model, golden_png, display_size=32)

    # Recompute from the ONNX features with numpy: logit_k = W_k · mean(F) + b_k
    x = preprocess(load_image(golden_png), IMAGE_SIZE)
    _, features = model.session.run(None, {"image": x})
    from cxr_core.bundle import load_head

    weight, bias = load_head(make_bundle(name="again"))
    logits = weight @ features[0].mean(axis=(1, 2)) + bias
    expected = 1 / (1 + np.exp(-logits))
    assert list(result.probabilities) == list(CLASSES)
    np.testing.assert_allclose(list(result.probabilities.values()), expected, rtol=1e-5)


def test_display_and_heatmaps_keep_the_aspect_ratio(
    make_bundle: BundleFactory, golden_png: Path
) -> None:
    result = run_inference(load_bundle(make_bundle()), golden_png, display_size=32)
    assert result.display.shape == (24, 32)  # golden_png is 48 x 64
    for heatmap in result.heatmaps.values():
        assert heatmap.shape == (24, 32)
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0


@pytest.mark.parametrize(
    ("shape", "expected"),
    [((1024, 1024), (512, 512)), ((100, 50), (100, 50)), ((2000, 1000), (512, 256))],
)
def test_display_shape(shape: tuple[int, int], expected: tuple[int, int]) -> None:
    assert display_shape(shape, 512) == expected


def test_undecodable_upload_is_a_bad_image(make_bundle: BundleFactory, tmp_path: Path) -> None:
    path = tmp_path / "broken.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"garbage" * 10)
    with pytest.raises(BadImage):
        run_inference(load_bundle(make_bundle()), path, display_size=32)


def test_to_png_is_8_bit_grayscale(tmp_path: Path) -> None:
    path = tmp_path / "out.png"
    path.write_bytes(to_png(np.array([[0.0, 0.5], [1.0, 2.0]], dtype=np.float32)))
    with Image.open(path) as img:
        assert img.mode == "L"
        assert np.asarray(img).tolist() == [[0, 128], [255, 255]]
