import math

import numpy as np
import pandas as pd
import pytest

from cxr_flows.evaluate import choose_thresholds, evaluate, render_html

CLASSES = ("A", "B")


def _data(n: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, (n, 2)).astype(np.float32)
    p = np.clip(0.35 * y + rng.random((n, 2)) * 0.65, 0, 1)
    meta = pd.DataFrame(
        {
            "patient_id": [str(i // 2) for i in range(n)],
            "patient_sex": rng.choice(["M", "F"], n),
            "view_position": ["PA"] * (n - 10) + ["AP"] * 10,
            "patient_age": pd.array(rng.integers(20, 90, n), dtype="Int64"),
        }
    )
    return y, p, meta


def test_youden_threshold_separates_separable_scores() -> None:
    y = np.array([[0], [0], [0], [1], [1]], dtype=np.float32)
    p = np.array([[0.1], [0.2], [0.3], [0.7], [0.9]])
    (t,) = choose_thresholds(y, p, ["A"]).values()
    assert 0.3 < t <= 0.7
    assert ((p[:, 0] >= t) == y[:, 0].astype(bool)).all()


def test_threshold_defaults_when_a_class_has_no_positives() -> None:
    y = np.zeros((4, 1), dtype=np.float32)
    assert choose_thresholds(y, np.full((4, 1), 0.2), ["A"]) == {"A": 0.5}


def test_report_structure_and_values() -> None:
    y, p, meta = _data()
    thresholds = choose_thresholds(y, p, CLASSES)
    report = evaluate(
        y, p, meta, CLASSES, thresholds,
        target_specificity=0.9, bootstrap_samples=50, min_subgroup_size=30,
    )  # fmt: skip
    assert report["n"] == 200 and report["n_patients"] == 100
    assert 0.5 < report["macro_auroc"] <= 1.0
    lo, hi = report["macro_auroc_ci95"]
    assert lo <= report["macro_auroc"] <= hi
    a = report["classes"]["A"]
    assert {"auroc", "auroc_ci95", "sensitivity_at_specificity_0.9", "ece", "threshold"} <= set(a)
    # AP has 10 images, below min_subgroup_size: reported with n but no metric.
    assert report["subgroups"]["view"]["AP"] == {"n": 10, "macro_auroc": None, "auroc": None}
    assert report["subgroups"]["view"]["PA"]["n"] == 190
    assert sum(g["n"] for g in report["subgroups"]["age"].values()) == 200


def test_undefined_metrics_are_null_not_nan() -> None:
    y, p, meta = _data(60)
    y[:, 1] = 0  # class B has no positives
    report = evaluate(
        y, p, meta, CLASSES, {"A": 0.5, "B": 0.5},
        target_specificity=0.9, bootstrap_samples=10, min_subgroup_size=10,
    )  # fmt: skip
    assert report["classes"]["B"]["auroc"] is None
    assert report["classes"]["B"]["sensitivity"] is None
    assert not math.isnan(report["macro_auroc"])  # nanmean over the defined class
    html = render_html("t", {"test": report})
    assert "<table>" in html and "n/a" in html


@pytest.mark.parametrize("seed", range(3))
def test_bootstrap_is_reproducible(seed: int) -> None:
    y, p, meta = _data(seed=seed)
    kwargs = dict(target_specificity=0.9, bootstrap_samples=20, min_subgroup_size=30, seed=7)
    a = evaluate(y, p, meta, CLASSES, {"A": 0.5, "B": 0.5}, **kwargs)  # type: ignore[arg-type]
    b = evaluate(y, p, meta, CLASSES, {"A": 0.5, "B": 0.5}, **kwargs)  # type: ignore[arg-type]
    assert a == b
