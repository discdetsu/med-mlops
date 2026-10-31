import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pydantic
import pytest

from cxr_flows.config import GateConfig, load_gate_config
from cxr_flows.gating import Scored, decide, render_html, version_tags

CLASSES = ("A", "B", "C")
N = 1200


def _cfg(**overrides: object) -> GateConfig:
    base = load_gate_config().model_dump() | {"bootstrap_samples": 200}
    return GateConfig.model_validate(base | overrides)


def _data(seed: int = 0) -> tuple[np.ndarray, pd.DataFrame, np.random.Generator]:
    rng = np.random.default_rng(seed)
    y = (rng.random((N, len(CLASSES))) < 0.3).astype(np.float32)
    meta = pd.DataFrame(
        {
            "patient_id": [str(i // 3) for i in range(N)],
            "patient_sex": np.where(np.arange(N) % 2 == 0, "M", "F"),
            "view_position": np.where(np.arange(N) < N // 2, "PA", "AP"),
            "patient_age": pd.array(rng.integers(20, 90, N), dtype="Int64"),
        }
    )
    return y, meta, rng


def _scores(y: np.ndarray, rng: np.random.Generator, signal: float) -> np.ndarray:
    """Probabilities whose separation grows with `signal`."""
    logits = signal * (2 * y - 1) + rng.normal(0, 1, y.shape)
    return 1 / (1 + np.exp(-logits))


def _checks(report: dict[str, object]) -> dict[str, bool]:
    return {c["name"]: c["passed"] for c in report["checks"]}  # type: ignore[attr-defined]


def _subgroup(report: dict[str, Any], attribute: str, value: str) -> dict[str, Any]:
    (match,) = [
        s for s in report["subgroups"] if (s["attribute"], s["value"]) == (attribute, value)
    ]
    return match


def test_the_bundled_gate_config_loads() -> None:
    cfg = load_gate_config()
    assert cfg.margin == 0.02
    assert not cfg.auto_promote


def test_gate_config_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "gate.yaml"
    path.write_text("margin: 0.02\nbootstrap_samples: 500\nmin_subgroup_size: 50\n"
                    "max_ece_increase: 0.01\nmargn: 0.1\n")  # fmt: skip
    with pytest.raises(pydantic.ValidationError):
        load_gate_config(str(path))


def test_a_better_challenger_passes() -> None:
    y, meta, rng = _data()
    champion = Scored("1", _scores(y, rng, 0.8), parity_passed=True)
    challenger = Scored("2", _scores(y, rng, 1.2), parity_passed=True)

    report = decide(_cfg(), y, meta, CLASSES, champion, challenger)

    assert report["passed"], report["checks"]
    assert report["macro_auroc"]["delta"] > 0
    lo, hi = report["macro_auroc"]["delta_ci"]
    assert lo < report["macro_auroc"]["delta"] < hi
    assert report["n_patients"] == N // 3
    assert [s["attribute"] for s in report["subgroups"]] == ["sex"] * 2 + ["view"] * 2 + ["age"] * 3
    assert [c["class"] for c in report["classes"]] == list(CLASSES)
    json.dumps(report, allow_nan=False)  # JSON-safe: stored in model_candidates


def test_the_same_model_passes_with_a_zero_difference() -> None:
    y, meta, rng = _data()
    p = _scores(y, rng, 1.0)
    report = decide(_cfg(), y, meta, CLASSES, Scored("1", p, True), Scored("2", p, True))
    assert report["passed"]
    assert report["macro_auroc"]["delta"] == 0.0
    assert report["macro_auroc"]["delta_ci"] == [0.0, 0.0]


def test_a_clearly_worse_challenger_fails_on_macro_auroc() -> None:
    y, meta, rng = _data()
    champion = Scored("1", _scores(y, rng, 1.2), True)
    challenger = Scored("2", _scores(y, rng, 0.6), True)

    report = decide(_cfg(), y, meta, CLASSES, champion, challenger)

    assert not report["passed"]
    assert not _checks(report)["macro_auroc"]


def test_a_regression_in_one_subgroup_fails_the_gate() -> None:
    """Better on PA, much worse on AP: the macro bound may pass, the AP subgroup must not."""
    y, meta, rng = _data()
    base = _scores(y, rng, 1.0)
    better = _scores(y, rng, 1.6)
    worse = _scores(y, rng, 0.3)
    ap = (meta["view_position"] == "AP").to_numpy()
    challenger = np.where(ap[:, None], worse, better)

    report = decide(_cfg(margin=0.05), y, meta, CLASSES, Scored("1", base, True),
                    Scored("2", challenger, True))  # fmt: skip

    assert not _checks(report)["subgroups"]
    assert not _subgroup(report, "view", "AP")["passed"]
    assert _subgroup(report, "view", "PA")["passed"]
    assert "view=AP" in report["checks"][2]["detail"]


def test_small_subgroups_are_reported_but_not_judged() -> None:
    y, meta, rng = _data()
    p = _scores(y, rng, 1.0)
    report = decide(_cfg(min_subgroup_size=N), y, meta, CLASSES, Scored("1", p, True),
                    Scored("2", p, True))  # fmt: skip
    ap = _subgroup(report, "view", "AP")
    assert ap["n"] == N // 2 and not ap["judged"] and ap["passed"]


def test_a_large_subgroup_without_a_delta_fails_the_gate() -> None:
    """No AP image has any finding, so no AUROC is defined there: the gate can't show the
    challenger is fine on 600 images, and must not pass them silently."""
    y, meta, rng = _data()
    ap = (meta["view_position"] == "AP").to_numpy()
    y[ap] = 0
    p = _scores(y, rng, 1.0)

    report = decide(_cfg(), y, meta, CLASSES, Scored("1", p, True), Scored("2", p, True))

    group = _subgroup(report, "view", "AP")
    assert group["delta"] is None and group["judged"] and not group["passed"]
    assert not _checks(report)["subgroups"] and not report["passed"]
    assert "view=AP (no ΔAUROC, n=600)" in report["checks"][2]["detail"]
    render_html(report)


def test_version_tags_carry_the_verdict() -> None:
    y, meta, rng = _data()
    p = _scores(y, rng, 1.0)
    report = decide(_cfg(), y, meta, CLASSES, Scored("1", p, True), Scored("2", p, True))

    tags = version_tags(report, "s3://artifacts/x.html")
    assert tags == {
        "gate_passed": "true",
        "gate_champion_version": "1",
        "gate_macro_delta_auroc": "+0.0000",
        "gate_report_uri": "s3://artifacts/x.html",
    }

    report["macro_auroc"]["delta"] = None  # no class with both positives and negatives
    assert version_tags(report, "s3://artifacts/x.html")["gate_macro_delta_auroc"] == "n/a"


def test_failed_parity_fails_the_gate_unless_not_required() -> None:
    y, meta, rng = _data()
    p = _scores(y, rng, 1.0)
    champion, challenger = Scored("1", p, True), Scored("2", p, parity_passed=False)

    assert not _checks(decide(_cfg(), y, meta, CLASSES, champion, challenger))["parity"]
    lenient = decide(_cfg(require_parity=False), y, meta, CLASSES, champion, challenger)
    assert lenient["passed"]


def test_worse_calibration_fails_the_ece_check() -> None:
    y, meta, rng = _data()
    p = _scores(y, rng, 1.0)
    # Same ranking, so AUROC is identical, but every probability is pushed towards 1.
    overconfident = p**0.2
    report = decide(_cfg(), y, meta, CLASSES, Scored("1", p, True),
                    Scored("2", overconfident, True))  # fmt: skip
    assert _checks(report)["macro_auroc"]
    assert not _checks(report)["ece"]
    assert report["ece"]["delta"] > 0.01


def test_mismatched_shapes_are_refused() -> None:
    y, meta, rng = _data()
    p = _scores(y, rng, 1.0)
    with pytest.raises(ValueError, match="probabilities"):
        decide(_cfg(), y, meta, CLASSES, Scored("1", p, True), Scored("2", p[:, :2], True))


def test_html_report_shows_the_verdict_and_escapes_names() -> None:
    y, meta, rng = _data()
    champion = Scored("1", _scores(y, rng, 1.2), True)
    challenger = Scored("2", _scores(y, rng, 0.6), True)
    report = decide(_cfg(), y, meta, ("<A>", "B", "C"), champion, challenger)

    page = render_html(report)

    assert "Gate FAILED" in page
    assert "version 2 vs @champion version 1" in page
    assert "&lt;A&gt;" in page and "<A>" not in page
    assert "Not a medical device" in page
