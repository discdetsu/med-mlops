"""Drift scoring and delayed AUROC on synthetic references and windows (no services)."""

import math
import uuid

import numpy as np
import pandas as pd
import pytest

from cxr_flows import drift
from cxr_flows.config import MonitorConfig, load_monitor_config

CLASSES = ("Effusion", "Mass")
DIM = 8


@pytest.fixture(scope="module")
def cfg() -> MonitorConfig:
    return load_monitor_config().model_copy(
        update={"min_window_size": 80, "window_size": 200, "min_feedback": 20,
                "bootstrap_samples": 200}
    )  # fmt: skip


def _population(n: int, rng: np.random.Generator, *, ap_share: float) -> pd.DataFrame:
    """Embeddings and probabilities that depend on the view, like real AP vs PA films."""
    ap = rng.random(n) < ap_share
    emb = rng.normal(0, 1, (n, DIM)) + np.where(ap, 1.0, 0.0)[:, None]
    logit = emb[:, :2] - 1
    return pd.DataFrame(
        {
            "view_position": np.where(ap, "AP", "PA"),
            "patient_sex": rng.choice(["M", "F"], n),
            "patient_age": rng.integers(20, 90, n).astype(float),
            **{f"prob_{c}": 1 / (1 + np.exp(-logit[:, k])) for k, c in enumerate(CLASSES)},
            **{f"emb_{i:04d}": emb[:, i] for i in range(DIM)},
        }
    )


def _rows(frame: pd.DataFrame, *, with_metadata: bool = True) -> pd.DataFrame:
    """The reference layout → what db.read_windows returns."""
    n = len(frame)
    emb = frame[[c for c in frame if c.startswith("emb_")]].to_numpy()
    blank = pd.Series([""] * n)
    return pd.DataFrame(
        {
            "job_id": [uuid.uuid4() for _ in range(n)],
            "model_version": "2",
            "created_at": pd.Timestamp("2026-11-01", tz="UTC"),
            "probabilities": [
                {c: float(frame[f"prob_{c}"].iloc[i]) for c in CLASSES} for i in range(n)
            ],
            "embedding": [list(map(float, e)) for e in emb],
            "patient_sex": frame["patient_sex"] if with_metadata else blank,
            "patient_age": frame["patient_age"].astype("Int64") if with_metadata else None,
            "view_position": frame["view_position"] if with_metadata else blank,
        }
    )


@pytest.fixture(scope="module")
def reference() -> pd.DataFrame:
    return _population(1500, np.random.default_rng(0), ap_share=0.37)


def test_window_frame_matches_the_reference_layout(reference: pd.DataFrame) -> None:
    rows = _rows(_population(5, np.random.default_rng(1), ap_share=0.5))
    rows.loc[0, "view_position"] = ""
    window = drift.window_frame(rows, reference)

    assert drift.embedding_columns(window) == drift.embedding_columns(reference)
    assert drift.classes_of(window) == list(CLASSES)
    assert window["view_position"].isna().tolist() == [True, False, False, False, False]
    assert window["patient_age"].dtype == "float64"


def test_window_frame_refuses_another_embedding_size(reference: pd.DataFrame) -> None:
    rows = _rows(_population(3, np.random.default_rng(1), ap_share=0.5))
    rows["embedding"] = [[0.0] * (DIM + 1)] * 3
    with pytest.raises(drift.DriftError, match="9 values, the reference has 8"):
        drift.window_frame(rows, reference)


def test_same_distribution_is_not_drift(reference: pd.DataFrame, cfg: MonitorConfig) -> None:
    window = drift.window_frame(
        _rows(_population(150, np.random.default_rng(2), ap_share=0.37)), reference
    )
    result = drift.compute_drift(reference, window, cfg)

    assert not result.detected, [s for s in result.scores if s.drifted]
    kinds = {(s.kind, s.feature) for s in result.scores}
    assert kinds == {
        ("embedding", "embedding"),
        ("prediction", "Effusion"),
        ("prediction", "Mass"),
        ("metadata", "view_position"),
        ("metadata", "patient_sex"),
        ("metadata", "patient_age"),
    }
    assert 0.3 < result.max_score("embedding") < cfg.embedding_threshold  # type: ignore[operator]


def test_ap_only_window_drifts_in_every_kind(reference: pd.DataFrame, cfg: MonitorConfig) -> None:
    window = drift.window_frame(
        _rows(_population(150, np.random.default_rng(3), ap_share=1.0)), reference
    )
    result = drift.compute_drift(reference, window, cfg)
    drifted = {(s.kind, s.feature) for s in result.scores if s.drifted}

    assert result.detected
    assert ("embedding", "embedding") in drifted
    assert ("metadata", "view_position") in drifted
    assert any(kind == "prediction" for kind, _ in drifted)
    assert ("metadata", "patient_sex") not in drifted  # sex didn't change


def test_metadata_is_judged_only_with_enough_known_values(
    reference: pd.DataFrame, cfg: MonitorConfig
) -> None:
    rows = _rows(_population(150, np.random.default_rng(4), ap_share=0.37), with_metadata=False)
    result = drift.compute_drift(reference, drift.window_frame(rows, reference), cfg)
    assert {s.kind for s in result.scores} == {"embedding", "prediction"}


def test_too_small_a_window_is_refused(reference: pd.DataFrame, cfg: MonitorConfig) -> None:
    window = drift.window_frame(
        _rows(_population(10, np.random.default_rng(5), ap_share=0.37)), reference
    )
    with pytest.raises(drift.DriftError, match="at least 80"):
        drift.compute_drift(reference, window, cfg)


def test_report_renders_to_html(reference: pd.DataFrame, cfg: MonitorConfig, tmp_path) -> None:  # type: ignore[no-untyped-def]
    window = drift.window_frame(
        _rows(_population(100, np.random.default_rng(6), ap_share=0.37)), reference
    )
    path = drift.compute_drift(reference, window, cfg).save_html(tmp_path / "r.html")
    assert path.read_text().lstrip().lower().startswith("<!doctype html")


def test_reference_sample_is_stable(reference: pd.DataFrame) -> None:
    a = drift.sample_reference(reference, 100, seed=0)
    b = drift.sample_reference(reference, 100, seed=0)
    pd.testing.assert_frame_equal(a, b)
    assert len(drift.sample_reference(reference, 10_000, seed=0)) == len(reference)


def _feedback(n: int, rng: np.random.Generator, signal: float) -> pd.DataFrame:
    y = rng.integers(0, 2, (n, len(CLASSES)))
    p = 1 / (1 + np.exp(-(signal * (y - 0.5) + rng.normal(0, 1, y.shape))))
    return pd.DataFrame(
        {
            "labels": [{c: int(y[i, k]) for k, c in enumerate(CLASSES)} for i in range(n)],
            "probabilities": [{c: float(p[i, k]) for k, c in enumerate(CLASSES)} for i in range(n)],
        }
    )


def test_delayed_auroc_is_not_reported_below_the_minimum(cfg: MonitorConfig) -> None:
    perf = drift.delayed_auroc(_feedback(19, np.random.default_rng(0), 3), CLASSES, cfg)
    assert not perf.reported
    assert perf.to_dict() == {
        "reported": False, "n": 19, "min_n": 20, "macro": None, "macro_ci": None, "per_class": {},
    }  # fmt: skip


def test_delayed_auroc_with_ci(cfg: MonitorConfig) -> None:
    perf = drift.delayed_auroc(_feedback(200, np.random.default_rng(0), 3), CLASSES, cfg)
    assert perf.reported and perf.macro is not None and perf.macro_ci is not None
    assert 0.8 < perf.macro < 1.0
    assert perf.macro_ci[0] < perf.macro < perf.macro_ci[1]
    assert set(perf.per_class or {}) == set(CLASSES)


def test_delayed_auroc_without_any_defined_class_is_not_reported(cfg: MonitorConfig) -> None:
    fb = _feedback(30, np.random.default_rng(0), 3)
    fb["labels"] = [dict.fromkeys(CLASSES, 0)] * 30  # every reviewer said "nothing"
    perf = drift.delayed_auroc(fb, CLASSES, cfg)
    assert not perf.reported and perf.n == 30


def test_fingerprint_ignores_order() -> None:
    ids = [uuid.uuid4() for _ in range(3)]
    assert drift.fingerprint(ids) == drift.fingerprint(reversed(ids))
    assert drift.fingerprint(ids) != drift.fingerprint(ids[:2])


def test_score_drifted_means_above_threshold() -> None:
    s = drift.Score("embedding", "embedding", "m", 100, 0.6, 0.6)
    assert not s.drifted
    assert drift.Score("embedding", "embedding", "m", 100, 0.61, 0.6).to_dict()["drifted"]
    assert math.isclose(s.to_dict()["score"], 0.6)
