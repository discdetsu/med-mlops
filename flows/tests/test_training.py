from pathlib import Path

import numpy as np
import pandas as pd

from cxr_flows import train as train_module
from cxr_flows.config import load_config
from cxr_flows.dataset import Split, load_split, merge_train, sample_train
from cxr_flows.reference import build_reference
from cxr_flows.schema import CLASSES
from cxr_flows.settings import Paths
from cxr_flows.training import predict, train_model


def _split(n: int, seed: int) -> Split:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, (n, len(CLASSES))).astype(np.float32)
    x = rng.standard_normal((n, 3, 64, 64)).astype(np.float32)
    x[:, :, :8, :8] += y[:, :1, None, None] * 2  # a learnable cue for class 0
    meta = pd.DataFrame(
        {
            "image_id": [f"{i:08d}_000" for i in range(n)],
            "patient_sex": "M",
            "patient_age": pd.array([50] * n, dtype="Int64"),
            "view_position": "PA",
        }
    )
    return Split(meta, x, y)


def test_one_epoch_trains_and_keeps_the_best_state() -> None:
    cfg = load_config("demo").model_copy(
        update={"pretrained": False, "epochs": 2, "image_size": 64, "batch_size": 8}
    )
    train, val = _split(32, 0), _split(16, 1)
    result = train_model(cfg, train, val, len(CLASSES))
    assert [h["epoch"] for h in result.history] == [1, 2]
    assert result.best_epoch in (1, 2)
    assert not result.model.training

    preds = predict(result.model, val.x, 8)
    assert preds.logits.shape == (16, len(CLASSES))
    assert preds.embeddings.shape == (16, 512)
    ref = build_reference(val.meta, preds, CLASSES)
    assert len(ref) == 16
    assert {f"prob_{c}" for c in CLASSES} <= set(ref.columns)
    assert ref.filter(like="emb_").shape[1] == 512


def test_tensor_cache_is_reused_and_matches_preprocess(
    tmp_path: Path, rows: object, images: object
) -> None:
    from cxr_core.preprocess import load_image, preprocess

    df = rows(4)  # type: ignore[operator]
    images(df, tmp_path / "ds")  # type: ignore[operator]
    split = load_split(df, tmp_path / "ds", "abc.dir", "train", 32, tmp_path / "cache", workers=1)
    expected = preprocess(load_image(tmp_path / "ds" / df["file"].iloc[0]), 32)[0]
    np.testing.assert_array_equal(split.x[0], expected)

    cached = next((tmp_path / "cache" / "tensors").iterdir()) / "train.npy"
    mtime = cached.stat().st_mtime_ns
    load_split(df, tmp_path / "ds", "abc.dir", "train", 32, tmp_path / "cache", workers=1)
    assert cached.stat().st_mtime_ns == mtime


def _cached(tmp_path: Path, df: pd.DataFrame, images: object, name: str) -> Split:
    images(df, tmp_path / name)  # type: ignore[operator]
    return load_split(df, tmp_path / name, f"{name}.dir", name, 32, tmp_path / "cache", workers=1)


def test_merge_train_drops_held_back_patients_and_adds_curated(
    tmp_path: Path, rows: object, images: object
) -> None:
    base = _cached(tmp_path, rows(4), images, "nih")  # type: ignore[operator]
    curated = rows(2, patient_offset=50)  # type: ignore[operator]
    curated["Mass"] = 1
    extra = _cached(tmp_path, curated, images, "curated")

    path = tmp_path / "cache" / "merged" / "m.npy"
    merged = merge_train(base, {"00000000"}, extra, path)  # patient 0 owns images 0 and 1

    assert list(merged.meta["image_id"]) == [
        "00000001_000", "00000001_001", "00000050_000", "00000050_001",
    ]  # fmt: skip
    np.testing.assert_array_equal(merged.x[0], base.x[2])
    np.testing.assert_array_equal(merged.x[2], extra.x[0])
    mass = CLASSES.index("Mass")
    assert merged.y[:, mass].tolist() == [0, 0, 1, 1]

    mtime = path.stat().st_mtime_ns
    merge_train(base, {"00000000"}, extra, path)
    assert path.stat().st_mtime_ns == mtime  # same members: cached


def test_sample_train_is_stable_and_ignores_row_order(rows: object) -> None:
    df = rows(20)  # type: ignore[operator]
    first = sample_train(df, 5, 0)
    assert list(first["image_id"]) == list(sample_train(df.iloc[::-1], 5, 0)["image_id"])
    assert len(sample_train(df, None, 0)) == 20 and len(sample_train(df, 99, 0)) == 20


def _pinned(curated: pd.DataFrame | None, md5: str = "cur.dir") -> train_module.PinnedData:
    md5s = {"data/nih": "nih.dir", "data/test_locked": "lock.dir"}
    empty = pd.DataFrame()
    return train_module.PinnedData("rev", md5s | {"data/curated": md5}, empty, empty, [], curated)


def test_recent_patients_never_train_and_curated_train_images_do(
    tmp_path: Path, rows: object, images: object
) -> None:
    paths = Paths(tmp_path)
    base = _cached(tmp_path, rows(4), images, "nih")  # type: ignore[operator]
    curated = rows(4, patient_offset=0).assign(label_source="reviewer")  # type: ignore[operator]
    curated["image_id"] = [f"upload-{i}" for i in range(4)]
    curated["file"] = [f"images/upload-{i}.png" for i in range(4)]
    curated["split"] = ["recent", "recent", "train", "train"]  # patient 0 is held back
    images(curated, paths.curated)  # type: ignore[operator]

    merged, recent = train_module._add_curated(paths, _pinned(curated), curated, base, 32)

    assert recent is not None
    assert list(recent.meta["image_id"]) == ["upload-0", "upload-1"]
    assert "00000000" not in set(merged.meta["patient_id"])  # its NIH images left too
    assert list(merged.meta["image_id"]) == [
        "00000001_000", "00000001_001", "upload-2", "upload-3",
    ]  # fmt: skip


def test_a_curated_set_with_only_recent_rows_still_holds_them_back(
    tmp_path: Path, rows: object, images: object
) -> None:
    paths = Paths(tmp_path)
    base = _cached(tmp_path, rows(4), images, "nih")  # type: ignore[operator]
    curated = rows(2).assign(label_source="reviewer", split="recent")  # type: ignore[operator]
    images(curated, paths.curated)  # type: ignore[operator]
    merged, recent = train_module._add_curated(paths, _pinned(curated), curated, base, 32)
    assert recent is not None and len(recent) == 2
    assert list(merged.meta["image_id"]) == ["00000001_000", "00000001_001"]
