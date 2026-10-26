"""Evaluation metrics shared by the train flow, the promotion gate, and the monitor flow.

Pure numpy so the gate's bootstrap can call them thousands of times without overhead.
"""

import math

import numpy as np
import numpy.typing as npt

# Inputs are converted with np.asarray, so anything array-like is accepted.
Labels = npt.ArrayLike
Scores = npt.ArrayLike


def binary_auroc(y_true: Labels, y_score: Scores) -> float:
    """AUROC via the Mann-Whitney U statistic; ties count half. NaN if one class is absent."""
    y = np.asarray(y_true).astype(bool).ravel()
    s = np.asarray(y_score, dtype=np.float64).ravel()
    n_pos = int(y.sum())
    n_neg = y.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return math.nan
    ranks = _average_ranks(s)
    u = ranks[y].sum() - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def per_class_auroc(y_true: Labels, y_score: Scores) -> npt.NDArray[np.float64]:
    """AUROC for each column of (N, K) multi-label arrays. NaN for columns with one class."""
    y, s = np.asarray(y_true), np.asarray(y_score)
    if y.shape != s.shape or y.ndim != 2:
        raise ValueError(f"expected matching (N, K) arrays, got {y.shape} and {s.shape}")
    return np.array([binary_auroc(y[:, k], s[:, k]) for k in range(y.shape[1])])


def sensitivity_at_specificity(y_true: Labels, y_score: Scores, specificity: float) -> float:
    """Highest sensitivity whose threshold keeps specificity >= `specificity`.

    A case is positive when score >= threshold. Keeping at least k = ceil(spec * n_neg)
    negatives strictly below the threshold leaves only positives scoring above the k-th
    lowest negative score.
    """
    if not 0.0 <= specificity <= 1.0:
        raise ValueError(f"specificity must be in [0, 1], got {specificity}")
    y = np.asarray(y_true).astype(bool).ravel()
    s = np.asarray(y_score, dtype=np.float64).ravel()
    pos, neg = s[y], np.sort(s[~y])
    if pos.size == 0 or neg.size == 0:
        return math.nan
    k = math.ceil(specificity * neg.size - 1e-12)
    if k == 0:
        return 1.0
    return float((pos > neg[k - 1]).mean())


def expected_calibration_error(y_true: Labels, y_score: Scores, n_bins: int = 15) -> float:
    """ECE with equal-width bins over [0, 1]. Multi-label input is pooled across classes."""
    y = np.asarray(y_true, dtype=np.float64).ravel()
    p = np.asarray(y_score, dtype=np.float64).ravel()
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: {np.shape(y_true)} vs {np.shape(y_score)}")
    if y.size == 0:
        return math.nan
    if p.min() < 0.0 or p.max() > 1.0:
        raise ValueError("scores must be probabilities in [0, 1]")

    # Bin i covers [i/n, (i+1)/n); the last bin also takes p == 1.
    bins = np.minimum((p * n_bins).astype(np.int64), n_bins - 1)
    counts = np.bincount(bins, minlength=n_bins)
    conf = np.bincount(bins, weights=p, minlength=n_bins)
    acc = np.bincount(bins, weights=y, minlength=n_bins)
    filled = counts > 0
    gap = np.abs(acc[filled] - conf[filled])  # per-bin |sum(y) - sum(p)| = n_b * |acc - conf|
    return float(gap.sum() / y.size)


def _average_ranks(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """1-based ranks with ties given their average rank."""
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    # Start index of each run of equal values, and run lengths.
    boundaries = np.flatnonzero(np.diff(sorted_vals)) + 1
    starts = np.concatenate(([0], boundaries))
    lengths = np.diff(np.concatenate((starts, [values.size])))
    avg = starts + (lengths + 1) / 2  # mean of ranks start+1 .. start+length
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.repeat(avg, lengths)
    return ranks
