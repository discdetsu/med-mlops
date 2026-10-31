"""The promotion gate's decision: may `@challenger` replace `@champion`? (ADR 0007)

Both versions have scored the same locked test set (promotion.py does the I/O). A paired
bootstrap over patients gives a CI for each AUROC difference, challenger - champion. The gate
passes when every check in `checks` passes:

- `parity`: the challenger passed E2E parity (ONNX vs PyTorch), when `require_parity` is on.
- `macro_auroc`: the lower CI bound of macro ΔAUROC is not below -margin (non-inferiority).
- `subgroups`: no subgroup with n >= `min_subgroup_size` has a macro ΔAUROC point estimate
  below -margin. Subgroup CIs are reported but not judged: they are wide enough that a
  non-inferiority test per subgroup would reject almost any retrain.
- `ece`: pooled ECE rises by at most `max_ece_increase`.

Passing makes the version a candidate; a person still approves it in Django admin.
"""

import html
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from cxr_core.metrics import expected_calibration_error, paired_bootstrap_auroc
from cxr_flows.config import GateConfig
from cxr_flows.evaluate import fmt, fmt_ci, html_page, subgroup_masks

Array = npt.NDArray[np.floating[Any]]


@dataclass(frozen=True)
class Scored:
    """One model version's probabilities on the gate's test set, rows in the same order."""

    version: str
    probabilities: Array  # (N, K)
    parity_passed: bool


def decide(
    cfg: GateConfig,
    y: Array,
    meta: pd.DataFrame,
    classes: Sequence[str],
    champion: Scored,
    challenger: Scored,
) -> dict[str, Any]:
    """The gate report: decision, checks, and every number they rest on (JSON-safe)."""
    n = len(meta)
    if y.shape != (n, len(classes)):
        raise ValueError(f"labels {y.shape} don't match {n} rows and {len(classes)} classes")
    for scored in (champion, challenger):
        if scored.probabilities.shape != y.shape:
            raise ValueError(
                f"version {scored.version}: probabilities {scored.probabilities.shape}, "
                f"labels {y.shape}"
            )

    masks = subgroup_masks(meta)
    groups = {"all": np.ones(n, dtype=bool)} | {
        _group_key(attr, value): m for attr, values in masks.items() for value, m in values.items()
    }
    paired = paired_bootstrap_auroc(
        y,
        champion.probabilities,
        challenger.probabilities,
        n_boot=cfg.bootstrap_samples,
        seed=cfg.seed,
        groups=groups,
        clusters=meta["patient_id"].to_numpy(),
    )
    overall = paired["all"]
    tail = 100 * (1 - cfg.confidence) / 2
    margin = cfg.margin

    def ci(values: Array) -> list[float] | None:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return None
        lo, hi = np.percentile(finite, [tail, 100 - tail])
        return [float(lo), float(hi)]

    macro_ci = ci(overall.boot_macro_delta)
    macro = {
        **_pair(overall.a, overall.b),
        "delta": _num(overall.macro_delta),
        "delta_ci": macro_ci,
    }
    # Lists, not dicts keyed by name: Postgres jsonb doesn't keep key order.
    per_class = [
        {
            "class": c,
            "champion": _num(overall.a[j]),
            "challenger": _num(overall.b[j]),
            "delta": _num(overall.delta[j]),
            "delta_ci": ci(overall.boot_delta[:, j]),
        }
        for j, c in enumerate(classes)
    ]

    subgroups: list[dict[str, Any]] = []
    for attr, values in masks.items():
        for value in values:
            g = paired[_group_key(attr, value)]
            delta = _num(g.macro_delta)
            judged = g.n >= cfg.min_subgroup_size and delta is not None
            subgroups.append(
                {
                    "attribute": attr,
                    "value": value,
                    "n": g.n,
                    **_pair(g.a, g.b),
                    "delta": delta,
                    "delta_ci": ci(g.boot_macro_delta),
                    "judged": judged,
                    "passed": (not judged) or (delta is not None and delta >= -margin),
                }
            )

    ece_champion = expected_calibration_error(y, champion.probabilities)
    ece_challenger = expected_calibration_error(y, challenger.probabilities)
    ece_increase = ece_challenger - ece_champion

    lower = macro_ci[0] if macro_ci else None
    failing = [
        f"{s['attribute']}={s['value']} ({s['delta']:+.4f}, n={s['n']})"
        for s in subgroups
        if not s["passed"]
    ]
    judged_count = sum(s["judged"] for s in subgroups)
    pct = f"{cfg.confidence:.0%}"
    checks = [
        {
            "name": "parity",
            "passed": challenger.parity_passed or not cfg.require_parity,
            "detail": (
                "E2E parity passed"
                if challenger.parity_passed
                else "E2E parity did not pass"
                + ("" if cfg.require_parity else " (not required: require_parity is off)")
            ),
        },
        {
            "name": "macro_auroc",
            "passed": lower is not None and lower >= -margin,
            "detail": (
                f"lower {pct} bound of macro ΔAUROC is {lower:+.4f}, needs ≥ {-margin:+.4f}"
                if lower is not None
                else "macro ΔAUROC has no CI: no class had both positives and negatives"
            ),
        },
        {
            "name": "subgroups",
            "passed": not failing,
            "detail": (
                f"below {-margin:+.4f}: " + ", ".join(failing)
                if failing
                else f"{judged_count} subgroups with n ≥ {cfg.min_subgroup_size} are all "
                f"within {-margin:+.4f}"
            ),
        },
        {
            "name": "ece",
            "passed": math.isfinite(ece_increase) and ece_increase <= cfg.max_ece_increase,
            "detail": (
                f"pooled ECE {ece_champion:.4f} → {ece_challenger:.4f} "
                f"({ece_increase:+.4f}, limit +{cfg.max_ece_increase:.4f})"
            ),
        },
    ]

    return {
        "passed": all(c["passed"] for c in checks),
        "champion_version": champion.version,
        "challenger_version": challenger.version,
        "n": n,
        "n_patients": int(meta["patient_id"].nunique()),
        "checks": checks,
        "macro_auroc": macro,
        "classes": per_class,
        "subgroups": subgroups,
        "ece": {
            "champion": _num(ece_champion),
            "challenger": _num(ece_challenger),
            "delta": _num(ece_increase),
        },
        "config": cfg.model_dump(),
    }


def render_html(report: dict[str, Any]) -> str:
    champion, challenger = report["champion_version"], report["challenger_version"]
    verdict = "PASSED" if report["passed"] else "FAILED"
    title = f"Promotion gate: version {challenger} vs @champion version {champion}"
    pct = f"{report['config']['confidence']:.0%}"

    check_rows = "".join(
        f"<tr><td>{html.escape(c['name'])}</td>{_verdict_cell(c['passed'])}"
        f"<td>{html.escape(c['detail'])}</td></tr>"
        for c in report["checks"]
    )
    m = report["macro_auroc"]
    class_rows = "".join(
        f"<tr><td>{html.escape(v['class'])}</td><td>{fmt(v['champion'])}</td>"
        f"<td>{fmt(v['challenger'])}</td><td>{fmt(v['delta'], sign=True)} "
        f"{fmt_ci(v['delta_ci'], sign=True)}</td></tr>"
        for v in report["classes"]
    )
    subgroup_rows = "".join(
        f"<tr><td>{html.escape(s['attribute'])}</td><td>{html.escape(s['value'])}</td>"
        f"<td>{s['n']}</td><td>{fmt(s['champion'])}</td><td>{fmt(s['challenger'])}</td>"
        f"<td>{fmt(s['delta'], sign=True)} {fmt_ci(s['delta_ci'], sign=True)}</td>"
        + (_verdict_cell(s["passed"]) if s["judged"] else "<td>not judged</td>")
        + "</tr>"
        for s in report["subgroups"]
    )
    ece = report["ece"]
    body = (
        f"<p class='{'pass' if report['passed'] else 'fail'}'>Gate {verdict}</p>"
        f"<p>Locked test set: n = {report['n']} images from {report['n_patients']} patients. "
        f"Δ = challenger - champion; {pct} CIs from a paired bootstrap over patients "
        f"({report['config']['bootstrap_samples']} resamples).</p>"
        "<h2>Checks</h2><table><tr><th>Check</th><th>Result</th><th>Detail</th></tr>"
        f"{check_rows}</table>"
        "<h2>AUROC</h2><table><tr><th>Class</th><th>Champion</th><th>Challenger</th>"
        f"<th>Δ ({pct} CI)</th></tr>"
        f"<tr><td><strong>macro</strong></td><td>{fmt(m['champion'])}</td>"
        f"<td>{fmt(m['challenger'])}</td><td>{fmt(m['delta'], sign=True)} "
        f"{fmt_ci(m['delta_ci'], sign=True)}</td></tr>{class_rows}</table>"
        "<h2>Subgroups (macro AUROC)</h2><table><tr><th>Group</th><th>Value</th><th>n</th>"
        f"<th>Champion</th><th>Challenger</th><th>Δ ({pct} CI)</th><th>Result</th></tr>"
        f"{subgroup_rows}</table>"
        f"<h2>Calibration</h2><p>Pooled ECE: champion {fmt(ece['champion'])}, "
        f"challenger {fmt(ece['challenger'])} ({fmt(ece['delta'], sign=True)}).</p>"
    )
    return html_page(title, body)


def _verdict_cell(passed: bool) -> str:
    return "<td class='pass'>pass</td>" if passed else "<td class='fail'>FAIL</td>"


def _group_key(attr: str, value: str) -> str:
    return f"{attr}={value}"


def _pair(a: Array, b: Array) -> dict[str, float | None]:
    """Macro AUROC of each model over the classes defined for both."""
    defined = np.isfinite(a) & np.isfinite(b)
    if not defined.any():
        return {"champion": None, "challenger": None}
    return {"champion": float(a[defined].mean()), "challenger": float(b[defined].mean())}


def _num(value: float | np.floating[Any]) -> float | None:
    v = float(value)
    return None if math.isnan(v) else v
