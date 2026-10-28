"""Drift reference for one model version: what the validation set looked like to that model.

One Parquet row per validation image: metadata (sex, age, view), predicted probabilities
(`prob_<class>`), and pooled embeddings (`emb_0000`…). The Phase 4 monitor flow compares
production windows against the `@champion` version's reference with Evidently.
"""

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from cxr_flows.training import Predictions

METADATA_COLUMNS = ("image_id", "patient_sex", "patient_age", "view_position")


def build_reference(
    meta: pd.DataFrame, predictions: Predictions, classes: Sequence[str]
) -> pd.DataFrame:
    if len(meta) != len(predictions.logits):
        raise ValueError(f"{len(meta)} metadata rows but {len(predictions.logits)} predictions")
    probs = predictions.probabilities
    emb = predictions.embeddings
    width = len(str(emb.shape[1] - 1))
    return pd.concat(
        [
            meta.loc[:, list(METADATA_COLUMNS)].reset_index(drop=True),
            pd.DataFrame(probs, columns=[f"prob_{c}" for c in classes]),
            pd.DataFrame(emb, columns=[f"emb_{i:0{max(width, 4)}d}" for i in range(emb.shape[1])]),
        ],
        axis=1,
    )


def write_reference(df: pd.DataFrame, path: Path) -> Path:
    df.to_parquet(path, index=False)
    return path
