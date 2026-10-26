import math

import numpy as np
import pytest

from cxr_core.metrics import (
    binary_auroc,
    expected_calibration_error,
    per_class_auroc,
    sensitivity_at_specificity,
)


def _brute_auroc(y: np.ndarray, s: np.ndarray) -> float:
    pos, neg = s[y == 1], s[y == 0]
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (pos.size * neg.size))


def _brute_sens_at_spec(y: np.ndarray, s: np.ndarray, target: float) -> float:
    pos, neg = s[y == 1], s[y == 0]
    best = 0.0
    for t in [*np.unique(s), np.inf, -np.inf]:
        if (neg < t).mean() >= target - 1e-12:
            best = max(best, float((pos >= t).mean()))
    return best


def _random_case(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, 200)
    s = np.round(rng.random(200) + 0.3 * y, 1)  # rounding forces many ties
    return y, s


class TestAuroc:
    def test_known_value(self) -> None:
        y = np.array([0, 0, 1, 1])
        s = np.array([0.1, 0.4, 0.35, 0.8])
        assert binary_auroc(y, s) == pytest.approx(0.75)

    def test_extremes_and_ties(self) -> None:
        y = np.array([0, 0, 1, 1])
        assert binary_auroc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
        assert binary_auroc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
        assert binary_auroc(y, np.full(4, 0.5)) == 0.5

    def test_single_class_is_nan(self) -> None:
        assert math.isnan(binary_auroc(np.ones(5, dtype=int), np.arange(5.0)))

    @pytest.mark.parametrize("seed", range(5))
    def test_matches_brute_force_with_ties(self, seed: int) -> None:
        y, s = _random_case(seed)
        assert binary_auroc(y, s) == pytest.approx(_brute_auroc(y, s))

    def test_per_class(self) -> None:
        y = np.array([[0, 1], [0, 1], [1, 1], [1, 1]])
        s = np.array([[0.1, 0.5], [0.2, 0.5], [0.8, 0.5], [0.9, 0.5]])
        out = per_class_auroc(y, s)
        assert out[0] == 1.0
        assert math.isnan(out[1])

    def test_per_class_rejects_shape_mismatch(self) -> None:
        with pytest.raises(ValueError):
            per_class_auroc(np.zeros((4, 2)), np.zeros((4, 3)))


class TestSensitivityAtSpecificity:
    def test_known_value(self) -> None:
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        s = np.array([0.1, 0.2, 0.3, 0.6, 0.4, 0.5, 0.7, 0.9])
        # spec >= 0.75 needs 3 of 4 negatives below t → t > 0.3 → positives 0.4..0.9 all kept
        assert sensitivity_at_specificity(y, s, 0.75) == 1.0
        # spec = 1 needs t > 0.6 → only 0.7 and 0.9
        assert sensitivity_at_specificity(y, s, 1.0) == 0.5

    def test_zero_specificity_is_full_sensitivity(self) -> None:
        y, s = _random_case(0)
        assert sensitivity_at_specificity(y, s, 0.0) == 1.0

    @pytest.mark.parametrize("seed", range(5))
    @pytest.mark.parametrize("target", [0.5, 0.8, 0.9, 0.95, 1.0])
    def test_matches_brute_force_with_ties(self, seed: int, target: float) -> None:
        y, s = _random_case(seed)
        assert sensitivity_at_specificity(y, s, target) == pytest.approx(
            _brute_sens_at_spec(y, s, target)
        )

    def test_rejects_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            sensitivity_at_specificity(np.array([0, 1]), np.array([0.1, 0.9]), 1.5)


class TestEce:
    def test_perfectly_confident_and_right_is_zero(self) -> None:
        y = np.array([0, 0, 1, 1])
        assert expected_calibration_error(y, y.astype(float)) == 0.0

    def test_overconfident(self) -> None:
        y = np.array([0, 1, 0, 1])
        assert expected_calibration_error(y, np.full(4, 0.8)) == pytest.approx(0.3)

    def test_multilabel_is_pooled(self) -> None:
        y = np.array([[0, 1], [1, 0]])
        p = np.full((2, 2), 0.8)
        assert expected_calibration_error(y, p) == pytest.approx(0.3)

    def test_probability_one_lands_in_last_bin(self) -> None:
        assert expected_calibration_error(np.array([1]), np.array([1.0]), n_bins=10) == 0.0

    def test_rejects_non_probabilities(self) -> None:
        with pytest.raises(ValueError):
            expected_calibration_error(np.array([0, 1]), np.array([-0.1, 1.2]))
