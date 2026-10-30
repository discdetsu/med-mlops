"""Loading serving bundles and swapping the worker's current model.

Only the main thread touches `ModelHolder`: the solo pool runs jobs there, and the
`model.events` consumer (events.py) runs in the same event loop between jobs. Assigning
`current` is the swap, so a job always finishes on the model it started with.
"""

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import onnxruntime as ort

from cxr_core.bundle import HEAD_FILE, MANIFEST_FILE, ONNX_FILE, Manifest, load_head
from cxr_core.inference import OnnxSession
from cxr_core.preprocess import PREPROCESS_VERSION
from cxr_core.registry import split_s3_uri
from cxr_worker.backend import BackendClient, ChampionInfo, NoChampion

logger = logging.getLogger(__name__)

# reference.parquet is for the monitor flow; serving doesn't need it.
SERVING_FILES = (MANIFEST_FILE, ONNX_FILE, HEAD_FILE)


class BundleError(Exception):
    pass


class ModelUnavailable(Exception):
    """No champion is loaded and none could be loaded."""


@dataclass(frozen=True)
class LoadedModel:
    manifest: Manifest
    session: OnnxSession
    head_weight: npt.NDArray[np.float32]

    @property
    def version(self) -> str:
        return self.manifest.model_version


def load_bundle(bundle_dir: Path) -> LoadedModel:
    manifest = Manifest.load(bundle_dir)
    if manifest.preprocess_version != PREPROCESS_VERSION:
        # Serving a model with different preprocessing than it was trained with gives
        # silently wrong probabilities; refuse instead.
        raise BundleError(
            f"version {manifest.model_version} needs preprocess_version "
            f"{manifest.preprocess_version}, this worker has {PREPROCESS_VERSION}"
        )
    weight, _ = load_head(bundle_dir)
    if weight.shape[0] != len(manifest.classes):
        raise BundleError(f"head has {weight.shape[0]} rows for {len(manifest.classes)} classes")
    session = ort.InferenceSession(str(bundle_dir / ONNX_FILE), providers=["CPUExecutionProvider"])
    return LoadedModel(manifest, session, weight)


def download_bundle(s3: Any, uri: str, dest: Path) -> Path:
    """Download the serving files into `dest`, appearing there only once complete."""
    bucket, prefix = split_s3_uri(uri)
    partial = dest.with_name(dest.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    for name in SERVING_FILES:
        s3.download_file(bucket, prefix + name, str(partial / name))
    shutil.rmtree(dest, ignore_errors=True)
    partial.rename(dest)
    return dest


class ModelHolder:
    def __init__(self, cache_dir: Path, s3: Any, backend: BackendClient) -> None:
        self.cache_dir = cache_dir
        self.s3 = s3
        self.backend = backend
        self.current: LoadedModel | None = None
        # Set when a swap fails, so the next job asks the registry again instead of serving
        # the old version until the worker reconnects.
        self.stale = False

    def load(self, info: ChampionInfo) -> LoadedModel:
        """Make `info` the current model; a no-op when it already is."""
        if self.current is not None and self.current.version == info.model_version:
            return self.current
        dest = self.cache_dir / info.model_name / info.model_version
        if not (dest / MANIFEST_FILE).exists():
            download_bundle(self.s3, info.bundle_uri, dest)
        model = load_bundle(dest)
        if model.version != info.model_version:
            raise BundleError(
                f"{info.bundle_uri} holds version {model.version}, expected {info.model_version}"
            )
        previous, self.current = self.current, model
        logger.info(
            "serving %s version %s (was %s)",
            info.model_name,
            model.version,
            previous.version if previous else "none",
        )
        self._prune(keep=dest)
        return model

    def sync(self) -> LoadedModel | None:
        """Load whatever the backend says is `@champion`. Keeps the current model on errors."""
        try:
            model = self.load(self.backend.champion())
        except NoChampion as exc:
            logger.warning("no model to serve yet: %s", exc)
        except Exception:
            self.stale = True
            logger.exception("loading the champion failed; still serving %s", self._serving())
        else:
            self.stale = False
            return model
        return self.current

    def switch(self, info: ChampionInfo) -> None:
        """Swap to `info` for a model event. On failure keep serving the current model and
        retry through the registry before the next job."""
        try:
            self.load(info)
        except Exception:
            self.stale = True
            logger.exception(
                "swapping to version %s failed; still serving %s",
                info.model_version,
                self._serving(),
            )

    def require(self) -> LoadedModel:
        if self.current is None or self.stale:
            self.sync()
        if self.current is None:
            raise ModelUnavailable("no @champion loaded; run `make seed-model`")
        return self.current

    def _serving(self) -> str:
        return self.current.version if self.current else "nothing"

    def _prune(self, keep: Path) -> None:
        for path in keep.parent.iterdir():
            if path != keep and path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
