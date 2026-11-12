"""The labeling loop's decisions, as pure functions (ADR 0009): which cases to send to Label
Studio, what a reviewer answered, where a new label belongs, and when to retrain.

The flows around them are `cxr_flows.monitor` (selection) and `cxr_flows.label_export` (export and
`should_retrain`). Nothing here imports torch: the monitor image doesn't have it.
"""

import hashlib
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from itertools import zip_longest
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from cxr_core.registry import REASON_DISAGREEMENT, REASON_DRIFT, REASON_RANDOM
from cxr_flows.config import LabelingConfig
from cxr_flows.drift import embedding_columns, fingerprint
from cxr_flows.labelstudio import FROM_NAME, NONE_CHOICE
from cxr_flows.nih import patient_split
from cxr_flows.schema import CURATED_RECENT, CURATED_TRAIN

# The deployment a retrain starts; `cxr_flows.deployments` serves it. Named here so the export flow
# doesn't have to import the training code.
TRAIN_DEPLOYMENT = "train/retrain"

# Patient-level, like the main split (ADR 0003), so a patient is never on both sides. A salt of its
# own, so this share is independent of the train/val/test buckets.
RECENT_SALT = "cxr-recent-v1"
RECENT_PERCENT = 20


@dataclass(frozen=True)
class Case:
    """A prediction to send to a reviewer."""

    sha256: str
    job_id: str
    image_key: str
    model_version: str
    reason: str


def curated_split(patient_id: str) -> str:
    """Where a reviewer label for this patient belongs.

    `test_locked` here means the label is refused: the patient's hash bucket is reserved for the
    locked test set, and `partition_labels` quarantines the row. `val` patients only ever go to
    the recent slice, so reviewer labels never change the validation set behind thresholds and the
    drift reference, nor put a validation patient into training.
    """
    bucket = patient_split(patient_id)
    if bucket == "test_locked":
        return bucket
    if bucket == "val":
        return CURATED_RECENT
    digest = hashlib.sha256(f"{RECENT_SALT}:{patient_id}".encode()).hexdigest()
    return CURATED_RECENT if int(digest, 16) % 100 < RECENT_PERCENT else CURATED_TRAIN


def is_labelable(patient_id: object) -> bool:
    """Whether a label for this case could ever enter the data. Without a patient ID the locked
    test set can't be protected; a test_locked patient's label would be refused. Either way a
    reviewer's time would be wasted."""
    return bool(patient_id) and curated_split(str(patient_id)) != "test_locked"


def disagreements(feedback: pd.DataFrame) -> pd.DataFrame:
    """Feedback rows whose labels differ from the model's thresholded predictions, newest first.

    `feedback` has one row per job: `labels`, `probabilities`, and `thresholds`, each {class: ...}.
    """
    if feedback.empty:
        return feedback

    def differs(row: Any) -> bool:
        return any(
            bool(row.labels[c]) != (row.probabilities[c] >= row.thresholds[c]) for c in row.labels
        )

    mask = [differs(row) for row in feedback.itertuples()]
    return feedback[mask].sort_values("created_at", ascending=False)


def embedding_distance(
    embeddings: npt.NDArray[np.floating[Any]], reference: pd.DataFrame
) -> npt.NDArray[np.floating[Any]]:
    """Each embedding's distance from the reference's centroid, in reference standard deviations
    per dimension, so a feature with a wide range doesn't dominate."""
    ref = reference[embedding_columns(reference)].to_numpy(dtype=np.float64)
    spread = ref.std(axis=0)
    spread[spread == 0] = 1.0
    return np.asarray(np.linalg.norm((embeddings - ref.mean(axis=0)) / spread, axis=1))


def farthest(rows: pd.DataFrame, reference: pd.DataFrame, k: int) -> pd.DataFrame:
    """The `k` rows whose `embedding` is farthest from the reference, farthest first."""
    if rows.empty or k == 0:
        return rows.iloc[:0]
    emb = np.array([list(e) for e in rows["embedding"]], dtype=np.float64)
    order = np.argsort(-embedding_distance(emb, reference), kind="stable")
    return rows.iloc[order[:k]]


def random_sample(rows: pd.DataFrame, k: int) -> pd.DataFrame:
    """`k` rows, the same ones for the same window."""
    if rows.empty or k == 0:
        return rows.iloc[:0]
    seed = int(fingerprint(rows["job_id"])[:16], 16)
    return rows.sample(n=min(k, len(rows)), random_state=seed % 2**32)


@dataclass(frozen=True)
class NewWindow:
    """A version's window that this monitor pass judged for the first time."""

    rows: pd.DataFrame
    drifted: bool
    reference: pd.DataFrame


def select_cases(
    feedback: pd.DataFrame,
    windows: Sequence[NewWindow],
    known: Collection[str],
    cfg: LabelingConfig,
) -> list[Case]:
    """Disagreements, the top-k of drifted windows, and a random sample of each new window, without
    any file already known or picked, up to `max_tasks_per_pass`.

    The cap trims disagreements first, newest kept: they are selected again on the next pass, while
    the other two come from a window once. Only when drift and the sample alone exceed it does
    drift lose cases, each window's nearest first, and the random sample goes last, since it is
    what limits selection bias.
    """
    chosen: dict[str, Case] = {}

    def usable(rows: pd.DataFrame) -> pd.DataFrame:
        if rows.empty:
            return rows
        keep = [
            is_labelable(p) and s not in known and s not in chosen
            for p, s in zip(rows["patient_id"], rows["sha256"], strict=True)
        ]
        return rows[keep].drop_duplicates("sha256")

    def add(rows: pd.DataFrame, reason: str) -> list[Case]:
        cases = [
            Case(str(r.sha256), str(r.job_id), str(r.image_key), str(r.model_version), reason)
            for r in rows.itertuples()
        ]
        chosen.update({c.sha256: c for c in cases})
        return cases

    disagreeing = add(usable(disagreements(feedback)), REASON_DISAGREEMENT)
    drift = _interleave(
        [
            add(farthest(usable(w.rows), w.reference, cfg.top_k_drift), REASON_DRIFT)
            for w in windows
            if w.drifted
        ]
    )
    sample = _interleave(
        [add(random_sample(usable(w.rows), cfg.random_k), REASON_RANDOM) for w in windows]
    )

    cap = cfg.max_tasks_per_pass
    sample = sample[:cap]
    drift = drift[: cap - len(sample)]
    return disagreeing[: cap - len(sample) - len(drift)] + drift + sample


def _interleave(groups: Sequence[list[Case]]) -> list[Case]:
    """Every group's first, then every group's second, so trimming the end treats windows alike."""
    return [c for rank in zip_longest(*groups) for c in rank if c is not None]


def parse_answer(
    annotations: Sequence[Mapping[str, Any]], classes: Sequence[str]
) -> dict[str, int] | None:
    """A task's latest finished annotation as {class: 0 or 1}.

    None when there is nothing usable yet: no annotation, only skipped ones, no answer at all, or
    "None of these" together with a finding. Such a task stays open for a reviewer to fix.
    """
    done = [a for a in annotations if not a.get("was_cancelled")]
    if not done:
        return None
    latest = max(done, key=lambda a: (str(a.get("updated_at") or a.get("created_at")), a["id"]))
    answers = [r for r in latest.get("result", []) if r.get("from_name") == FROM_NAME]
    if not answers:
        return None
    picked = set(answers[0].get("value", {}).get("choices", []))
    if not picked or (NONE_CHOICE in picked and len(picked) > 1):
        return None
    if picked - {NONE_CHOICE} - set(classes):
        return None  # the project's choices don't match this code's classes
    return {c: int(c in picked) for c in classes}


def image_id_for(sha256: str) -> str:
    """Curated images are named by content, so re-uploads of a file are one image."""
    return f"upload-{sha256[:16]}"


@dataclass(frozen=True)
class RetrainDecision:
    trigger: bool
    reason: str
    dataset_rev: str
    baseline_rev: str
    new_labels: int
    min_new_labels: int


def should_retrain(
    *,
    dataset_rev: str,
    baseline_rev: str,
    last_triggered_rev: str | None,
    new_labels: int,
    min_new_labels: int,
) -> RetrainDecision:
    """Retrain on a new dataset rev with at least `min_new_labels` curated training images that the
    newest model version (`baseline_rev`) didn't have. Drift is not an input, on purpose.

    Counting from the newest version, not from the last trigger, means a retrain that failed
    doesn't lose its labels; a rev is only ever triggered once, so it doesn't loop either.
    """

    def decide(trigger: bool, reason: str) -> RetrainDecision:
        return RetrainDecision(
            trigger, reason, dataset_rev, baseline_rev, new_labels, min_new_labels
        )

    if dataset_rev == baseline_rev:
        return decide(False, "the newest model version already trained on this rev")
    if dataset_rev == last_triggered_rev:
        return decide(False, "a retrain was already started for this rev")
    if new_labels < min_new_labels:
        return decide(False, f"{new_labels} new labels, at least {min_new_labels} needed")
    return decide(True, f"{new_labels} new labels on a new rev, at least {min_new_labels} needed")
