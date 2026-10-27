from pathlib import Path

import numpy as np
import pandas as pd

from cxr_flows.config import load_config
from cxr_flows.dataset import Split, load_split
from cxr_flows.reference import build_reference
from cxr_flows.schema import CLASSES
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
