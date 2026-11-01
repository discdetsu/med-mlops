import math

import numpy as np
import pytest

from cxr_core.metrics import (
    binary_auroc,
    expected_calibration_error,
    paired_bootstrap_auroc,
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


def _two_models(seed: int, n: int = 300, k: int = 3) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, (n, k))
    a = np.round(rng.random((n, k)) + 0.4 * y, 1)  # ties on purpose
    b = np.round(rng.random((n, k)) + 0.8 * y, 1)  # better than a
    return y, a, b


class TestPairedBootstrap:
    def test_point_estimates_match_binary_auroc(self) -> None:
        y, a, b = _two_models(0)
        result = paired_bootstrap_auroc(y, a, b, n_boot=0)["all"]
        np.testing.assert_allclose(result.a, per_class_auroc(y, a))
        np.testing.assert_allclose(result.b, per_class_auroc(y, b))
        assert result.n == len(y)
        assert result.boot_a.shape == (0, 3)

    def test_each_resample_is_the_auroc_of_the_resampled_rows(self) -> None:
        """Re-draw the generator's resamples and score them the slow way."""
        y, a, b = _two_models(1, n=80, k=2)
        result = paired_bootstrap_auroc(y, a, b, n_boot=5, seed=7, chunk=2)["all"]

        rng = np.random.default_rng(7)
        for start in range(0, 5, 2):
            rows = min(2, 5 - start)
            draws = rng.integers(0, 80, (rows, 80))
            for r in range(rows):
                idx = draws[r]
                row = start + r
                np.testing.assert_allclose(result.boot_a[row], per_class_auroc(y[idx], a[idx]))
                np.testing.assert_allclose(result.boot_b[row], per_class_auroc(y[idx], b[idx]))

    def test_groups_score_their_own_cases(self) -> None:
        y, a, b = _two_models(2)
        mask = np.arange(len(y)) % 3 == 0
        result = paired_bootstrap_auroc(y, a, b, n_boot=0, groups={"third": mask})
        assert set(result) == {"third"}
        assert result["third"].n == mask.sum()
        np.testing.assert_allclose(result["third"].a, per_class_auroc(y[mask], a[mask]))

    def test_identical_models_differ_by_exactly_zero(self) -> None:
        y, a, _ = _two_models(3)
        result = paired_bootstrap_auroc(y, a, a, n_boot=50)["all"]
        assert result.macro_delta == 0.0
        np.testing.assert_array_equal(result.boot_macro_delta, np.zeros(50))

    def test_a_better_model_has_a_positive_interval(self) -> None:
        y, a, b = _two_models(4)
        result = paired_bootstrap_auroc(y, a, b, n_boot=200)["all"]
        lo, hi = np.percentile(result.boot_macro_delta, [2.5, 97.5])
        assert 0 < lo < result.macro_delta < hi

    def test_cluster_resampling_keeps_clusters_whole(self) -> None:
        """With two cases per cluster that always travel together, every case weight is even
        when each cluster's cases are duplicated rows: AUROC equals that of one copy each."""
        y, a, b = _two_models(5, n=60, k=2)
        y2, a2, b2 = np.repeat(y, 2, axis=0), np.repeat(a, 2, axis=0), np.repeat(b, 2, axis=0)
        clusters = np.repeat(np.arange(60), 2)
        single = paired_bootstrap_auroc(y, a, b, n_boot=20, seed=3)["all"]
        paired = paired_bootstrap_auroc(y2, a2, b2, n_boot=20, seed=3, clusters=clusters)["all"]
        np.testing.assert_allclose(paired.boot_a, single.boot_a)
        np.testing.assert_allclose(paired.boot_b, single.boot_b)

    def test_a_class_missing_from_a_resample_is_nan(self) -> None:
        y = np.array([[1], [0], [0], [0]])
        s = np.array([[0.9], [0.1], [0.2], [0.3]])
        result = paired_bootstrap_auroc(y, s, s, n_boot=200, seed=0)["all"]
        missing = ~np.isfinite(result.boot_a[:, 0])
        assert missing.any()  # P(no positive drawn) = (3/4)^4 ≈ 0.32
        assert np.isnan(result.boot_macro_delta[missing]).all()

    def test_rejects_mismatched_shapes(self) -> None:
        with pytest.raises(ValueError, match="matching"):
            paired_bootstrap_auroc(np.zeros((3, 2)), np.zeros((3, 2)), np.zeros((3, 1)), n_boot=1)
        with pytest.raises(ValueError, match="mask shape"):
            paired_bootstrap_auroc(
                np.zeros((3, 1)), np.zeros((3, 1)), np.zeros((3, 1)), n_boot=1, groups={"g": [1]}
            )
