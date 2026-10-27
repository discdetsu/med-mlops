"""Model inputs for training and evaluation, preprocessed once and cached on disk.

Every image goes through `cxr_core.preprocess` from the raw file, the same call serving
makes. The cache key is the dataset path's DVC md5 + PREPROCESS_VERSION + image size, so a
new data version or a preprocessing change never reuses stale tensors.
"""

import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from numpy.lib.format import open_memmap

from cxr_core.preprocess import PREPROCESS_VERSION, load_image, preprocess
from cxr_flows.schema import CLASSES


@dataclass(frozen=True)
class Split:
    """Canonical metadata rows and their inputs and labels, in the same order."""

    meta: pd.DataFrame
    x: npt.NDArray[np.float32]  # (N, 3, S, S), memory-mapped
    y: npt.NDArray[np.float32]  # (N, K)

    def __len__(self) -> int:
        return len(self.meta)

    def head(self, n: int) -> Split:
        return Split(self.meta.iloc[:n].reset_index(drop=True), self.x[:n], self.y[:n])


def load_split(
    meta: pd.DataFrame,
    dataset_dir: Path,
    dvc_md5: str,
    name: str,
    image_size: int,
    cache_root: Path,
    workers: int | None = None,
) -> Split:
    meta = meta.sort_values("image_id", ignore_index=True)
    key = f"{dvc_md5.removesuffix('.dir')[:16]}-p{PREPROCESS_VERSION}-s{image_size}"
    cache = cache_root / "tensors" / key
    x_path, ids_path = cache / f"{name}.npy", cache / f"{name}.ids"
    ids = "\n".join(meta["image_id"])

    if not (x_path.exists() and ids_path.exists() and ids_path.read_text() == ids):
        _build(meta, dataset_dir, image_size, cache, x_path, workers)
        ids_path.write_text(ids)

    x = np.load(x_path, mmap_mode="r")
    y = meta[list(CLASSES)].to_numpy(dtype=np.float32)
    return Split(meta, x, y)


def _build(
    meta: pd.DataFrame,
    dataset_dir: Path,
    image_size: int,
    cache: Path,
    x_path: Path,
    workers: int | None,
) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    tmp = x_path.with_suffix(".tmp.npy")
    out = open_memmap(
        tmp, mode="w+", dtype=np.float32, shape=(len(meta), 3, image_size, image_size)
    )
    files = [dataset_dir / str(f) for f in meta["file"]]
    with ProcessPoolExecutor(max_workers=workers or os.cpu_count()) as pool:
        for i, arr in enumerate(pool.map(_load, files, [image_size] * len(files), chunksize=16)):
            out[i] = arr
    out.flush()
    del out
    shutil.move(tmp, x_path)


def _load(path: Path, image_size: int) -> npt.NDArray[np.float32]:
    arr: npt.NDArray[np.float32] = preprocess(load_image(path), image_size)[0]
    return arr
