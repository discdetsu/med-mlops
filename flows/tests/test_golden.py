from pathlib import Path
from typing import Any

import pandas as pd

from cxr_flows.golden import build_golden, pick_golden, read_manifest


def test_pick_and_build(tmp_path: Path, rows: Any, images: Any) -> None:
    locked = rows(8, split="test_locked")
    locked.loc[3, "Cardiomegaly"] = 1
    locked.loc[5, "Effusion"] = 1
    locked.loc[1, "Effusion"] = 1
    images(locked, tmp_path / "locked")

    picks = pick_golden(locked)
    assert list(picks["image_id"]) == [
        locked.loc[3, "image_id"],
        locked.loc[1, "image_id"],
        locked.loc[0, "image_id"],
    ]
    manifest = build_golden(picks, tmp_path / "locked", tmp_path / "golden")
    pd.testing.assert_frame_equal(manifest, read_manifest(tmp_path / "golden"))
    assert len(manifest) == 9
    assert (manifest["photometric"] == "MONOCHROME1").sum() == 3
    assert all((tmp_path / "golden" / f).exists() for f in manifest["file"])
