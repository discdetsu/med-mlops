"""Thresholds, evaluation reports, and their JSON + HTML rendering.

Thresholds are chosen on the validation set only. Metrics are reported on the locked test
set, and on the recent slice when one exists (Phase 5). Reports carry n and bootstrap CIs
because NIH labels are noisy and the demo test set is small.
"""

import html
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from cxr_core.metrics import (
    binary_auroc,
    expected_calibration_error,
    per_class_auroc,
    sensitivity_at_specificity,
)

Array = npt.NDArray[np.floating[Any]]

AGE_BINS = ((0, 40, "<40"), (40, 60, "40-59"), (60, 200, "60+"))


def choose_thresholds(y_val: Array, p_val: Array, classes: Sequence[str]) -> dict[str, float]:
    """Per-class threshold maximizing Youden's J (sensitivity + specificity - 1) on val.

    A case is positive when probability >= threshold, matching `sensitivity_at_specificity`.
    """
    return {c: _youden_threshold(y_val[:, k], p_val[:, k]) for k, c in enumerate(classes)}


def _youden_threshold(y: Array, p: Array) -> float:
    positive = y.astype(bool)
    candidates = np.unique(p)
    if positive.all() or not positive.any():
        return 0.5
    # For each candidate t: sensitivity = P(p >= t | pos), specificity = P(p < t | neg).
    pos, neg = np.sort(p[positive]), np.sort(p[~positive])
    sens = 1.0 - np.searchsorted(pos, candidates, side="left") / pos.size
    spec = np.searchsorted(neg, candidates, side="left") / neg.size
    return float(candidates[int(np.argmax(sens + spec - 1.0))])


def evaluate(
    y: Array,
    p: Array,
    meta: pd.DataFrame,
    classes: Sequence[str],
    thresholds: Mapping[str, float],
    *,
    target_specificity: float,
    bootstrap_samples: int,
    min_subgroup_size: int,
    seed: int = 0,
) -> dict[str, Any]:
    """Per-class and macro metrics with bootstrap CIs, plus subgroup AUROCs (sex, age, view)."""
    rng = np.random.default_rng(seed)
    aurocs = per_class_auroc(y, p)
    boot = _bootstrap_aurocs(y, p, bootstrap_samples, rng)

    per_class = {}
    for k, c in enumerate(classes):
        yk, pk = y[:, k].astype(bool), p[:, k]
        predicted = pk >= thresholds[c]
        per_class[c] = {
            "n_positive": int(yk.sum()),
            "auroc": _num(aurocs[k]),
            "auroc_ci95": _ci(boot[:, k]) if boot.size else None,
            f"sensitivity_at_specificity_{target_specificity:g}": _num(
                sensitivity_at_specificity(yk, pk, target_specificity)
            ),
            "threshold": thresholds[c],
            "sensitivity": _num(predicted[yk].mean() if yk.any() else math.nan),
            "specificity": _num((~predicted[~yk]).mean() if (~yk).any() else math.nan),
            "ece": _num(expected_calibration_error(yk, pk)),
        }

    return {
        "n": len(y),
        "n_patients": int(meta["patient_id"].nunique()),
        "macro_auroc": _num(np.nanmean(aurocs)),
        "macro_auroc_ci95": _ci(np.nanmean(boot, axis=1)) if boot.size else None,
        "ece": _num(expected_calibration_error(y, p)),
        "classes": per_class,
        "subgroups": _subgroups(y, p, meta, classes, min_subgroup_size),
    }


def _subgroups(
    y: Array, p: Array, meta: pd.DataFrame, classes: Sequence[str], min_size: int
) -> dict[str, dict[str, Any]]:
    age = meta["patient_age"].astype("Float64")
    groups: dict[str, dict[str, npt.NDArray[np.bool_]]] = {
        "sex": {s: (meta["patient_sex"] == s).to_numpy() for s in ("M", "F")},
        "view": {v: (meta["view_position"] == v).to_numpy() for v in ("PA", "AP")},
        "age": {
            label: ((age >= lo) & (age < hi)).fillna(False).to_numpy(dtype=bool)
            for lo, hi, label in AGE_BINS
        },
    }
    out: dict[str, dict[str, Any]] = {}
    for name, masks in groups.items():
        out[name] = {}
        for value, mask in masks.items():
            n = int(mask.sum())
            if n < min_size:
                out[name][value] = {"n": n, "macro_auroc": None, "auroc": None}
                continue
            aurocs = per_class_auroc(y[mask], p[mask])
            out[name][value] = {
                "n": n,
                "macro_auroc": _num(np.nanmean(aurocs) if np.isfinite(aurocs).any() else math.nan),
                "auroc": {c: _num(aurocs[k]) for k, c in enumerate(classes)},
            }
    return out


def _bootstrap_aurocs(y: Array, p: Array, samples: int, rng: np.random.Generator) -> Array:
    """(samples, K) AUROCs over patient-agnostic resamples of rows. NaN where a class vanished."""
    if samples == 0:
        return np.empty((0, y.shape[1]))
    n = len(y)
    out = np.empty((samples, y.shape[1]))
    for b in range(samples):
        idx = rng.integers(0, n, n)
        out[b] = [binary_auroc(y[idx, k], p[idx, k]) for k in range(y.shape[1])]
    return out


def _ci(values: Array) -> list[float] | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return [float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))]


def _num(value: float | np.floating[Any]) -> float | None:
    """JSON-safe float: NaN (undefined metric, e.g. no positives) becomes null."""
    v = float(value)
    return None if math.isnan(v) else v


def render_html(title: str, reports: Mapping[str, Mapping[str, Any]]) -> str:
    """Minimal standalone HTML: one section per evaluated set."""
    sections = []
    for name, report in reports.items():
        rows = []
        for c, m in report["classes"].items():
            sens_key = next(k for k in m if k.startswith("sensitivity_at_specificity_"))
            rows.append(
                "<tr>"
                f"<td>{html.escape(c)}</td><td>{m['n_positive']}</td>"
                f"<td>{_fmt(m['auroc'])} {_fmt_ci(m['auroc_ci95'])}</td>"
                f"<td>{_fmt(m[sens_key])}</td><td>{_fmt(m['threshold'])}</td>"
                f"<td>{_fmt(m['sensitivity'])}</td><td>{_fmt(m['specificity'])}</td>"
                f"<td>{_fmt(m['ece'])}</td>"
                "</tr>"
            )
        sens_label = sens_key.removeprefix("sensitivity_at_specificity_") if rows else ""
        subgroup_rows = [
            f"<tr><td>{html.escape(group)}</td><td>{html.escape(value)}</td>"
            f"<td>{s['n']}</td><td>{_fmt(s['macro_auroc'])}</td></tr>"
            for group, values in report["subgroups"].items()
            for value, s in values.items()
        ]
        sections.append(
            f"<h2>{html.escape(name)}</h2>"
            f"<p>n = {report['n']} images from {report['n_patients']} patients · "
            f"macro AUROC {_fmt(report['macro_auroc'])} {_fmt_ci(report['macro_auroc_ci95'])} · "
            f"ECE {_fmt(report['ece'])}</p>"
            "<table><tr><th>Class</th><th>Positives</th><th>AUROC (95% CI)</th>"
            f"<th>Sens @ spec {sens_label}</th><th>Threshold</th><th>Sens</th><th>Spec</th>"
            f"<th>ECE</th></tr>{''.join(rows)}</table>"
            "<h3>Subgroups</h3><table><tr><th>Group</th><th>Value</th><th>n</th>"
            f"<th>Macro AUROC</th></tr>{''.join(subgroup_rows)}</table>"
        )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title><style>"
        "body{font-family:system-ui,sans-serif;margin:2rem;max-width:60rem}"
        "table{border-collapse:collapse;margin-bottom:1.5rem}"
        "td,th{border:1px solid #ccc;padding:.3rem .6rem;text-align:right}"
        "td:first-child,th:first-child{text-align:left}"
        "</style></head><body>"
        f"<h1>{html.escape(title)}</h1>"
        "<p><strong>Not a medical device.</strong> Research demo on NIH ChestX-ray14 "
        "NLP-mined labels.</p>"
        f"{''.join(sections)}</body></html>\n"
    )


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _fmt_ci(ci: Sequence[float] | None) -> str:
    return "" if ci is None else f"({ci[0]:.3f} to {ci[1]:.3f})"
