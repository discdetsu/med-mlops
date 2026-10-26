import numpy as np
import pytest

from cxr_core.cam import class_activation_map


def _features() -> np.ndarray:
    """2 channels on a 4x4 grid: channel 0 fires top-left, channel 1 bottom-right."""
    f = np.zeros((2, 4, 4), dtype=np.float32)
    f[0, 0, 0] = 1.0
    f[1, 3, 3] = 1.0
    return f


WEIGHTS = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]], dtype=np.float32)


@pytest.mark.parametrize(("class_index", "quadrant"), [(0, (0, 0)), (1, (1, 1))])
def test_heatmap_peaks_where_the_class_evidence_is(
    class_index: int, quadrant: tuple[int, int]
) -> None:
    heatmap = class_activation_map(_features(), WEIGHTS, class_index, (64, 64))
    row, col = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    assert (row // 32, col // 32) == quadrant


def test_output_shape_dtype_and_range() -> None:
    heatmap = class_activation_map(_features(), WEIGHTS, 0, (48, 32))
    assert heatmap.shape == (48, 32)
    assert heatmap.dtype == np.float32
    assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0
    assert heatmap.max() > 0.5


def test_only_negative_evidence_gives_zeros() -> None:
    heatmap = class_activation_map(_features(), WEIGHTS, 2, (16, 16))
    assert not heatmap.any()


def test_unclipped_map_decomposes_the_gap_logit() -> None:
    """With GAP + linear head, mean of the CAM equals the logit minus bias."""
    rng = np.random.default_rng(0)
    features = rng.random((8, 7, 7), dtype=np.float32)
    weights = rng.random((3, 8), dtype=np.float32)  # positive, so clipping is a no-op
    logits = weights @ features.mean(axis=(1, 2))
    for k in range(3):
        raw = np.tensordot(weights[k], features, axes=1)
        assert raw.mean() == pytest.approx(logits[k], rel=1e-5)
        heatmap = class_activation_map(features, weights, k, (7, 7))
        np.testing.assert_allclose(heatmap, raw / raw.max(), atol=1e-5)


def test_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError):
        class_activation_map(_features(), np.zeros((3, 5), dtype=np.float32), 0, (8, 8))
    with pytest.raises(ValueError):
        class_activation_map(_features()[None], WEIGHTS, 0, (8, 8))
    with pytest.raises(IndexError):
        class_activation_map(_features(), WEIGHTS, 3, (8, 8))
