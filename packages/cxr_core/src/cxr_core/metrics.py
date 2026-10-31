"""Evaluation metrics shared by the train flow, the promotion gate, and the monitor flow.

Pure numpy so the gate's bootstrap can call them thousands of times without overhead.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass

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


@dataclass(frozen=True)
class PairedAuroc:
    """Per-class AUROC of two models scored on the same cases: point estimates and bootstrap
    replicates drawn with the same resamples for both, so their differences are paired."""

    n: int  # cases in the group
    a: npt.NDArray[np.float64]  # (K,)
    b: npt.NDArray[np.float64]  # (K,)
    boot_a: npt.NDArray[np.float64]  # (B, K)
    boot_b: npt.NDArray[np.float64]  # (B, K)

    @property
    def delta(self) -> npt.NDArray[np.float64]:
        """b - a per class."""
        return self.b - self.a

    @property
    def boot_delta(self) -> npt.NDArray[np.float64]:
        return self.boot_b - self.boot_a

    @property
    def macro_delta(self) -> float:
        """Macro (mean over classes defined in both) AUROC of b minus that of a."""
        return _nanmean(self.delta)

    @property
    def boot_macro_delta(self) -> npt.NDArray[np.float64]:
        """(B,) macro ΔAUROC per resample; NaN where no class was defined."""
        d = self.boot_delta
        defined = np.isfinite(d)
        counts = defined.sum(axis=1)
        sums = np.where(defined, d, 0.0).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(counts > 0, sums / np.maximum(counts, 1), math.nan)


def paired_bootstrap_auroc(
    y_true: Labels,
    score_a: Scores,
    score_b: Scores,
    *,
    n_boot: int,
    seed: int = 0,
    groups: Mapping[str, npt.ArrayLike] | None = None,
    clusters: npt.ArrayLike | None = None,
    chunk: int = 64,
) -> dict[str, PairedAuroc]:
    """Paired bootstrap of per-class AUROC for models a and b on the same (N, K) labels.

    Every resample is applied to both models, so the spread of b - a reflects only how the
    models differ, not how the test set was drawn. With `clusters` (e.g. patient IDs), whole
    clusters are resampled, because images of one patient are not independent. `groups` maps
    a name to a boolean mask over the N cases; each group is evaluated on the same resamples
    (the cases of a resample that fall in the group). Without `groups` there is one, `all`.

    Each resample is a vector of case weights (how often each case was drawn), so a model's
    scores are sorted once and every resample costs one cumulative sum instead of a sort.
    """
    y = np.asarray(y_true).astype(bool)
    a = np.asarray(score_a, dtype=np.float64)
    b = np.asarray(score_b, dtype=np.float64)
    if y.ndim != 2 or a.shape != y.shape or b.shape != y.shape:
        raise ValueError(f"expected matching (N, K) arrays, got {y.shape}, {a.shape}, {b.shape}")
    n, k = y.shape
    if n == 0:
        raise ValueError("no cases")
    if n_boot < 0 or chunk < 1:
        raise ValueError(f"n_boot must be >= 0 and chunk >= 1, got {n_boot} and {chunk}")

    masks = {"all": np.ones(n, dtype=bool)} if groups is None else {}
    for name, mask in (groups or {}).items():
        m = np.asarray(mask, dtype=bool)
        if m.shape != (n,):
            raise ValueError(f"group {name!r}: mask shape {m.shape}, expected ({n},)")
        masks[name] = m

    if clusters is None:
        cluster_of = np.arange(n)
    else:
        c = np.asarray(clusters)
        if c.shape != (n,):
            raise ValueError(f"clusters shape {c.shape}, expected ({n},)")
        _, cluster_of = np.unique(c, return_inverse=True)
    n_clusters = int(cluster_of.max()) + 1

    # Sorted once per (group, class, model); a resample is then a gather and a cumulative sum.
    ranked = {
        name: [
            (_RankedScores(y[:, j], a[:, j], m), _RankedScores(y[:, j], b[:, j], m))
            for j in range(k)
        ]
        for name, m in masks.items()
    }
    ones = np.ones((1, n))
    points = {
        name: (
            np.array([ra.auroc(ones)[0] for ra, _ in pairs]),
            np.array([rb.auroc(ones)[0] for _, rb in pairs]),
        )
        for name, pairs in ranked.items()
    }

    boot = {name: (np.empty((n_boot, k)), np.empty((n_boot, k))) for name in masks}
    rng = np.random.default_rng(seed)
    for start in range(0, n_boot, chunk):
        rows = min(chunk, n_boot - start)
        draws = rng.integers(0, n_clusters, (rows, n_clusters))
        # Times each cluster was drawn, per resample row: one bincount over offset indices.
        offsets = draws + n_clusters * np.arange(rows)[:, None]
        counts = np.bincount(offsets.ravel(), minlength=rows * n_clusters)
        w = counts.reshape(rows, n_clusters)[:, cluster_of].astype(np.float64)
        for name, pairs in ranked.items():
            out_a, out_b = boot[name]
            for j, (ra, rb) in enumerate(pairs):
                out_a[start : start + rows, j] = ra.auroc(w)
                out_b[start : start + rows, j] = rb.auroc(w)

    return {name: PairedAuroc(int(masks[name].sum()), *points[name], *boot[name]) for name in masks}


class _RankedScores:
    """One class's labels and scores within one group, prepared once for weighted AUROC on
    many resamples: negatives sorted by score, and for each positive the number of negatives
    scoring below it and below-or-tied."""

    def __init__(
        self, y: npt.NDArray[np.bool_], s: npt.NDArray[np.float64], mask: npt.NDArray[np.bool_]
    ) -> None:
        members = np.flatnonzero(mask)
        self.pos = members[y[members]]
        neg = members[~y[members]]
        self.neg = neg[np.argsort(s[neg], kind="mergesort")]
        neg_scores = s[self.neg]
        self.below = np.searchsorted(neg_scores, s[self.pos], side="left")
        self.below_or_tied = np.searchsorted(neg_scores, s[self.pos], side="right")

    def auroc(self, w: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """(R, N) case weights in original order → (R,) weighted AUROC, ties counting half.

        With C the cumulative negative weight in score order, a positive whose score has
        `lo` negatives strictly below it and `hi` below or tied wins C[lo] + (C[hi] - C[lo]) / 2
        comparisons, i.e. (C[lo] + C[hi]) / 2.
        """
        rows = w.shape[0]
        if self.pos.size == 0 or self.neg.size == 0:
            return np.full(rows, math.nan)
        c = np.zeros((rows, self.neg.size + 1))
        np.cumsum(w[:, self.neg], axis=1, out=c[:, 1:])
        wp = w[:, self.pos]
        wins = (wp * (c[:, self.below] + c[:, self.below_or_tied])).sum(axis=1) / 2
        n_pos, n_neg = wp.sum(axis=1), c[:, -1]
        with np.errstate(invalid="ignore", divide="ignore"):
            out: npt.NDArray[np.float64] = wins / (n_pos * n_neg)
        out[(n_pos == 0) | (n_neg == 0)] = math.nan
        return out


def _nanmean(values: npt.NDArray[np.float64]) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else math.nan


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
