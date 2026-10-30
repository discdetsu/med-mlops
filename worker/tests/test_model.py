from pathlib import Path

import pytest

from cxr_worker.model import BundleError, ModelHolder, ModelUnavailable, load_bundle

from .conftest import BundleFactory, FakeS3
from .fakes import FakeBackend, info


def holder_with(
    tmp_path: Path, s3: FakeS3, backend: FakeBackend, make_bundle: BundleFactory, *versions: str
) -> ModelHolder:
    for v in versions:
        s3.add_bundle(info(v).bundle_uri, make_bundle(v))
    return ModelHolder(tmp_path / "cache", s3, backend)  # type: ignore[arg-type]


def test_bundle_for_other_preprocessing_is_refused(make_bundle: BundleFactory) -> None:
    with pytest.raises(BundleError, match="preprocess_version"):
        load_bundle(make_bundle(preprocess_version="0"))


def test_sync_loads_the_champion_once(
    tmp_path: Path, s3: FakeS3, make_bundle: BundleFactory
) -> None:
    holder = holder_with(tmp_path, s3, FakeBackend(info("1")), make_bundle, "1")
    model = holder.sync()
    assert model is not None
    assert model.version == "1"
    assert holder.sync() is model  # same version: no reload
    # reference.parquet is not downloaded; the worker doesn't need it
    assert sorted(p.name for p in (tmp_path / "cache/cxr-classifier/1").iterdir()) == [
        "head.npz",
        "manifest.json",
        "model.onnx",
    ]


def test_load_swaps_to_a_new_version_and_prunes_the_old_one(
    tmp_path: Path, s3: FakeS3, make_bundle: BundleFactory
) -> None:
    holder = holder_with(tmp_path, s3, FakeBackend(info("1")), make_bundle, "1", "2")
    first = holder.require()
    second = holder.load(info("2"))

    assert holder.current is second
    assert (first.version, second.version) == ("1", "2")
    assert not (tmp_path / "cache/cxr-classifier/1").exists()


def test_a_bundle_whose_manifest_disagrees_is_refused(
    tmp_path: Path, s3: FakeS3, make_bundle: BundleFactory
) -> None:
    s3.add_bundle(info("5").bundle_uri, make_bundle("4"))
    holder = ModelHolder(tmp_path / "cache", s3, FakeBackend())  # type: ignore[arg-type]
    with pytest.raises(BundleError, match="expected 5"):
        holder.load(info("5"))
    assert holder.current is None


def test_sync_keeps_serving_when_the_new_bundle_is_missing(
    tmp_path: Path, s3: FakeS3, make_bundle: BundleFactory
) -> None:
    backend = FakeBackend(info("1"))
    holder = holder_with(tmp_path, s3, backend, make_bundle, "1")
    holder.sync()
    backend.champion_info = info("9")  # not in S3

    model = holder.sync()
    assert model is not None
    assert model.version == "1"


def test_require_without_a_champion_raises(tmp_path: Path, s3: FakeS3) -> None:
    holder = ModelHolder(tmp_path / "cache", s3, FakeBackend())  # type: ignore[arg-type]
    with pytest.raises(ModelUnavailable):
        holder.require()
