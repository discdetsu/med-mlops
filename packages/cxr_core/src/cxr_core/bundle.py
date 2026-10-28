"""Serving bundle: everything one model version needs at inference time, plus its manifest.

The train flow writes it, uploads it to `s3://artifacts/models/<name>/<version>/`, and logs
it to MLflow; `bootstrap_champion.py` packs it as the seed bundle. The worker reads it, so
this module stays stdlib + numpy.

    manifest.json      classes, image size, thresholds, preprocess version, lineage
    model.onnx         inputs `image` (N, 3, S, S); outputs `logits` (N, K), `features` (N, C, h, w)
    head.npz           `weight` (K, C) and `bias` (K,) of the linear head, for CAM
    reference.parquet  validation-set embeddings, predictions, and metadata for drift
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

BUNDLE_FORMAT = 1
MANIFEST_FILE = "manifest.json"
ONNX_FILE = "model.onnx"
HEAD_FILE = "head.npz"
REFERENCE_FILE = "reference.parquet"
BUNDLE_FILES = (MANIFEST_FILE, ONNX_FILE, HEAD_FILE, REFERENCE_FILE)


@dataclass(frozen=True)
class Manifest:
    model_name: str
    model_version: str
    architecture: str
    classes: tuple[str, ...]
    image_size: int
    preprocess_version: str
    # Per-class decision thresholds chosen on the validation set. They belong to the model
    # version, not to serving config.
    thresholds: dict[str, float]
    # Where the version came from: dataset_rev, git_commit, mlflow_run_id, ...
    lineage: dict[str, str] = field(default_factory=dict)
    bundle_format: int = BUNDLE_FORMAT

    def __post_init__(self) -> None:
        if set(self.thresholds) != set(self.classes):
            raise ValueError(
                f"thresholds {sorted(self.thresholds)} don't match classes {list(self.classes)}"
            )

    def save(self, bundle_dir: Path) -> Path:
        path = bundle_dir / MANIFEST_FILE
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")
        return path

    @classmethod
    def load(cls, bundle_dir: Path) -> Manifest:
        data: dict[str, Any] = json.loads((bundle_dir / MANIFEST_FILE).read_text())
        if data.get("bundle_format") != BUNDLE_FORMAT:
            raise ValueError(
                f"{bundle_dir}: bundle_format {data.get('bundle_format')!r}, "
                f"this code reads {BUNDLE_FORMAT}"
            )
        data["classes"] = tuple(data["classes"])
        return cls(**data)


def save_head(
    bundle_dir: Path, weight: npt.NDArray[np.float32], bias: npt.NDArray[np.float32]
) -> Path:
    path = bundle_dir / HEAD_FILE
    np.savez(path, weight=weight.astype(np.float32), bias=bias.astype(np.float32))
    return path


def load_head(bundle_dir: Path) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    with np.load(bundle_dir / HEAD_FILE) as npz:
        return npz["weight"].astype(np.float32), npz["bias"].astype(np.float32)
