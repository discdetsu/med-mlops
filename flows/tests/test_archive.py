"""--from-archive: the full Kaggle archive, or an extracted copy, as the image source."""

import sys
import types
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from cxr_flows import nih


def _fake_archive(root: Path, files: list[str]) -> Path:
    (root / nih.DATA_ENTRY_FILE).parent.mkdir(parents=True, exist_ok=True)
    (root / nih.DATA_ENTRY_FILE).write_text("csv")
    for i in range(1, 13):
        (root / f"images_{i:03d}" / "images").mkdir(parents=True, exist_ok=True)
    for f in files:
        (root / f).write_bytes(b"png:" + f.encode())
    return root


def _no_network(*args: object, **kwargs: object) -> Path:
    raise AssertionError("per-file Kaggle download used despite an archive")


def test_parse_count() -> None:
    assert nih.parse_count("all") is None
    assert nih.parse_count("12") == 12
    with pytest.raises(ValueError):
        nih.parse_count("-1")


def test_check_archive_root(tmp_path: Path) -> None:
    root = _fake_archive(tmp_path / "a", [])
    assert nih.check_archive_root(root) == root
    (root / "images_007" / "images").rmdir()
    with pytest.raises(FileNotFoundError, match="images_007"):
        nih.check_archive_root(root)


def test_download_images_links_from_the_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(nih, "_kaggle_file", _no_network)
    rows = pd.DataFrame(
        {
            "kaggle_path": ["images_001/images/a.png", "images_012/images/b.png"],
            "file": ["images/a.png", "images/b.png"],
        }
    )
    archive = _fake_archive(tmp_path / "archive", list(rows["kaggle_path"]))
    dest = tmp_path / "nih"

    assert nih.download_images(rows, dest, tmp_path / "staging", 2, archive) == 2
    assert (dest / "images/b.png").read_bytes() == b"png:images_012/images/b.png"
    assert nih.download_images(rows, dest, tmp_path / "staging", 2, archive) == 0
    assert nih.fetch_data_entry(tmp_path / "staging", archive) == archive / nih.DATA_ENTRY_FILE


def test_fetch_archive_uses_kagglehub_cache_and_remembers_the_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "kagglehub"
    calls: list[dict[str, Any]] = []

    def dataset_download(handle: str, **kwargs: Any) -> str:
        import os

        calls.append({"handle": handle, "cache": os.environ["KAGGLEHUB_CACHE"], **kwargs})
        return str(_fake_archive(cache / "datasets/nih/versions/3", []))

    monkeypatch.setitem(
        sys.modules, "kagglehub", types.SimpleNamespace(dataset_download=dataset_download)
    )
    monkeypatch.delenv("KAGGLEHUB_CACHE", raising=False)
    monkeypatch.setattr(nih.shutil, "disk_usage", lambda _: types.SimpleNamespace(free=10**12))

    root = nih.fetch_archive(cache)
    # Whole dataset, kagglehub's own cache, and no output_dir: the archive stays outside the
    # extraction folder, so an interrupted download can resume.
    assert calls == [{"handle": nih.KAGGLE_HANDLE, "cache": str(cache)}]
    assert nih.fetch_archive(cache) == root
    assert len(calls) == 1  # second call answered from the marker, no network


def test_fetch_archive_refuses_without_disk_space(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(nih.shutil, "disk_usage", lambda _: types.SimpleNamespace(free=10**9))
    with pytest.raises(RuntimeError, match="GB free"):
        nih.fetch_archive(tmp_path / "kagglehub")


def test_files_from_a_user_folder_are_copied_not_linked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(nih, "_kaggle_file", _no_network)
    rows = pd.DataFrame({"kaggle_path": ["images_001/images/a.png"], "file": ["images/a.png"]})
    archive = _fake_archive(tmp_path / "archive", list(rows["kaggle_path"]))
    nih.download_images(rows, tmp_path / "copied", tmp_path / "s", 1, archive, link=False)
    nih.download_images(rows, tmp_path / "linked", tmp_path / "s", 1, archive, link=True)
    source = (archive / "images_001/images/a.png").stat()
    assert (tmp_path / "copied/images/a.png").stat().st_ino != source.st_ino
    assert (tmp_path / "linked/images/a.png").stat().st_ino == source.st_ino
