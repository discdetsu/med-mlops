"""Drift and delayed AUROC for the monitor flow, as pure functions over DataFrames.

A window is one model version's recent predictions, shaped like that version's drift reference
(`cxr_flows.reference`): `emb_*` embedding columns, `prob_<class>` probabilities, and the
metadata columns. Evidently scores three kinds of drift between the two:

- embedding: a domain classifier tries to tell window from reference embeddings; the score is
  its ROC AUC (0.5 when it can't);
- prediction: the normed Wasserstein distance per class probability;
- metadata: Jensen-Shannon distance for view and sex, normed Wasserstein for age, on the
  window's rows that have the field (PNG uploads without form fields have none).

Every score is "higher means more drift" and is compared with a threshold from monitor.yaml.
"""

import hashlib
import math
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandas.core.dtypes.common as _pandas_dtypes
from pandas.api.types import pandas_dtype

from cxr_core.metrics import bootstrap_auroc, percentile_ci
from cxr_flows.config import MonitorConfig

# Evidently 0.7.23 (the latest release) imports `infer_dtype_from_object`, which pandas 3
# removed, at import time. Only one legacy test class calls it, and none of the metrics here
# do; restore the pandas 2 behaviour (dtype → its scalar type) so the import succeeds. Drop
# this once Evidently supports pandas 3. tests/test_drift.py covers the metrics used.
if not hasattr(_pandas_dtypes, "infer_dtype_from_object"):
    _pandas_dtypes.infer_dtype_from_object = lambda dtype: pandas_dtype(dtype).type  # type: ignore[attr-defined]

with warnings.catch_warnings():
    # Import-time noise from Evidently and plotly about pandas 4, numpy.core, and invalid
    # escape sequences; none of it affects the metrics used here.
    warnings.simplefilter("ignore")
    from evidently import DataDefinition, Dataset, Report
    from evidently.legacy.metrics.data_drift.embedding_drift_methods import (
        model as domain_classifier,
    )
    from evidently.metrics import EmbeddingsDrift, ValueDrift

EMBEDDING_PREFIX = "emb_"
PROBABILITY_PREFIX = "prob_"
CATEGORICAL_METADATA = ("view_position", "patient_sex")
NUMERICAL_METADATA = ("patient_age",)
METADATA = (*CATEGORICAL_METADATA, *NUMERICAL_METADATA)
_EMBEDDINGS = "embedding"


class DriftError(ValueError):
    """The window can't be compared with the reference (e.g. a different embedding size)."""


@dataclass(frozen=True)
class Score:
    kind: str  # embedding | prediction | metadata
    feature: str  # embedding, the class name, or the metadata column
    method: str
    n: int  # window rows the score is based on
    score: float
    threshold: float

    @property
    def drifted(self) -> bool:
        return self.score > self.threshold

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "drifted": self.drifted}


@dataclass(frozen=True)
class DriftResult:
    scores: list[Score]
    n_window: int
    n_reference: int
    snapshot: Any  # Evidently snapshot, for the HTML report

    @property
    def detected(self) -> bool:
        return any(s.drifted for s in self.scores)

    def max_score(self, kind: str) -> float | None:
        values = [s.score for s in self.scores if s.kind == kind]
        return max(values) if values else None

    def save_html(self, path: Path) -> Path:
        self.snapshot.save_html(str(path))
        return path


def classes_of(reference: pd.DataFrame) -> list[str]:
    columns = [str(c) for c in reference.columns]
    return [c.removeprefix(PROBABILITY_PREFIX) for c in columns if c.startswith(PROBABILITY_PREFIX)]


def embedding_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(str(c) for c in frame.columns if str(c).startswith(EMBEDDING_PREFIX))


def window_frame(rows: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Predictions as read by `db.read_predictions` → the reference's column layout.

    `rows` has `embedding` (list of floats), `probabilities` ({class: p}), and the metadata
    columns, with "" or None where a field is unknown.
    """
    emb_cols = embedding_columns(reference)
    classes = classes_of(reference)
    emb = np.array([list(e) for e in rows["embedding"]], dtype=np.float32).reshape(len(rows), -1)
    if len(rows) and emb.shape[1] != len(emb_cols):
        raise DriftError(
            f"embeddings have {emb.shape[1]} values, the reference has {len(emb_cols)}"
        )
    try:
        probs = [[float(p[c]) for c in classes] for p in rows["probabilities"]]
    except KeyError as exc:
        raise DriftError(f"a prediction has no probability for {exc}") from None
    meta = pd.DataFrame(
        {
            "view_position": _blank_to_na(rows["view_position"]),
            "patient_sex": _blank_to_na(rows["patient_sex"]),
            "patient_age": pd.to_numeric(rows["patient_age"], errors="coerce").astype("float64"),
        }
    )
    return pd.concat(
        [
            meta.reset_index(drop=True),
            pd.DataFrame(
                np.array(probs, dtype=np.float64).reshape(len(rows), len(classes)),
                columns=[PROBABILITY_PREFIX + c for c in classes],
            ),
            pd.DataFrame(emb, columns=emb_cols),
        ],
        axis=1,
    )


def sample_reference(reference: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """At most `n` rows, the same ones on every call with the same seed."""
    if len(reference) <= n:
        return reference.reset_index(drop=True)
    return reference.sample(n, random_state=seed).reset_index(drop=True)


def compute_drift(reference: pd.DataFrame, window: pd.DataFrame, cfg: MonitorConfig) -> DriftResult:
    """Score `window` (from `window_frame`) against `reference` (already sampled)."""
    if len(window) < cfg.min_window_size:
        raise DriftError(f"{len(window)} predictions, at least {cfg.min_window_size} needed")
    emb_cols = embedding_columns(reference)
    if embedding_columns(window) != emb_cols:
        raise DriftError("window and reference embedding columns differ")
    classes = classes_of(reference)
    prob_cols = [PROBABILITY_PREFIX + c for c in classes]

    planned: list[tuple[Any, Score]] = [
        (
            EmbeddingsDrift(
                embeddings_name=_EMBEDDINGS,
                drift_method=domain_classifier(threshold=cfg.embedding_threshold),
            ),
            Score("embedding", "embedding", "domain classifier ROC AUC", len(window), math.nan,
                  cfg.embedding_threshold),
        )
    ]  # fmt: skip
    for c, col in zip(classes, prob_cols, strict=True):
        planned.append(
            (
                ValueDrift(column=col, method="wasserstein", threshold=cfg.prediction_threshold),
                Score("prediction", c, "wasserstein", len(window), math.nan,
                      cfg.prediction_threshold),
            )
        )  # fmt: skip
    thresholds = cfg.metadata_thresholds.model_dump()
    for col in METADATA:
        n = int(window[col].notna().sum())
        if n < cfg.min_window_size or reference[col].notna().sum() == 0:
            continue  # too few uploads carried this field to judge it
        method = "jensenshannon" if col in CATEGORICAL_METADATA else "wasserstein"
        planned.append(
            (
                ValueDrift(column=col, method=method, threshold=thresholds[col]),
                Score("metadata", col, method, n, math.nan, thresholds[col]),
            )
        )

    definition = DataDefinition(
        numerical_columns=[*NUMERICAL_METADATA, *prob_cols, *emb_cols],
        categorical_columns=list(CATEGORICAL_METADATA),
        embeddings={_EMBEDDINGS: emb_cols},
    )
    columns = [*METADATA, *prob_cols, *emb_cols]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # sklearn / scipy chatter on small windows
        snapshot = Report([metric for metric, _ in planned]).run(
            Dataset.from_pandas(_typed(window[columns]), data_definition=definition),
            Dataset.from_pandas(_typed(reference[columns]), data_definition=definition),
        )
    values = [float(m["value"]) for m in snapshot.dict()["metrics"]]
    if len(values) != len(planned):
        raise DriftError(f"Evidently returned {len(values)} metrics for {len(planned)} asked")
    scores = [
        Score(s.kind, s.feature, s.method, s.n, v, s.threshold)
        for (_, s), v in zip(planned, values, strict=True)
    ]
    return DriftResult(scores, len(window), len(reference), snapshot)


@dataclass(frozen=True)
class DelayedAuroc:
    n: int
    min_n: int
    macro: float | None = None
    macro_ci: tuple[float, float] | None = None
    per_class: Mapping[str, Mapping[str, Any]] | None = None

    @property
    def reported(self) -> bool:
        return self.macro is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "reported": self.reported,
            "n": self.n,
            "min_n": self.min_n,
            "macro": self.macro,
            "macro_ci": list(self.macro_ci) if self.macro_ci else None,
            "per_class": dict(self.per_class or {}),
        }


def delayed_auroc(
    feedback: pd.DataFrame, classes: Sequence[str], cfg: MonitorConfig
) -> DelayedAuroc:
    """AUROC of the version's probabilities against reviewer labels, with a bootstrap CI.

    `feedback` has one row per job: `labels` ({class: 0/1}) and `probabilities` ({class: p}).
    Below `min_feedback` rows nothing is reported; a class without both a positive and a
    negative label has no AUROC and is left out of the macro.
    """
    n = len(feedback)
    if n < cfg.min_feedback:
        return DelayedAuroc(n, cfg.min_feedback)
    y = np.array([[int(lab[c]) for c in classes] for lab in feedback["labels"]])
    p = np.array([[float(pr[c]) for c in classes] for pr in feedback["probabilities"]])
    result = bootstrap_auroc(y, p, n_boot=cfg.bootstrap_samples, seed=cfg.seed)
    if not math.isfinite(result.macro):
        return DelayedAuroc(n, cfg.min_feedback)
    per_class = {
        c: {
            "auroc": _finite(result.auroc[k]),
            "ci": _ci(percentile_ci(result.boot[:, k], cfg.confidence)),
            "positives": int(y[:, k].sum()),
        }
        for k, c in enumerate(classes)
    }
    return DelayedAuroc(
        n,
        cfg.min_feedback,
        macro=result.macro,
        macro_ci=percentile_ci(result.boot_macro, cfg.confidence),
        per_class=per_class,
    )


def fingerprint(job_ids: Iterable[object]) -> str:
    """Identifies a window by its members, so an unchanged window isn't scored twice."""
    digest = hashlib.sha256()
    for job_id in sorted(str(j) for j in job_ids):
        digest.update(job_id.encode() + b"\n")
    return digest.hexdigest()


def _blank_to_na(values: pd.Series) -> pd.Series:
    return values.astype("object").where(values.notna() & (values.astype(str) != ""), None)


def _typed(frame: pd.DataFrame) -> pd.DataFrame:
    """Categorical metadata as plain strings (NaN where unknown) on both sides, so Evidently
    sees the same categories."""
    out = frame.copy()
    for col in CATEGORICAL_METADATA:
        out[col] = out[col].astype("object").astype("str")  # pandas 3 keeps None as NaN
    out["patient_age"] = pd.to_numeric(out["patient_age"], errors="coerce").astype("float64")
    return out


def _finite(value: float) -> float | None:
    v = float(value)
    return v if math.isfinite(v) else None


def _ci(ci: tuple[float, float] | None) -> list[float] | None:
    return list(ci) if ci else None
