import hashlib
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from django.test import Client
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST

import cxr_backend.registry.api as registry_api
import cxr_backend.registry.seed as seed_module
from cxr_backend.registry.champion import Champion, current_champion
from cxr_backend.registry.seed import SeedError, extract, fetch, seed
from cxr_core.bundle import BUNDLE_FILES, Manifest
from cxr_core.messaging import ChampionChanged

from .conftest import FakeS3

MANIFEST = Manifest(
    model_name="cxr-classifier",
    model_version="2",
    architecture="resnet18",
    classes=("A", "B"),
    image_size=128,
    preprocess_version="1",
    thresholds={"A": 0.1, "B": 0.2},
    lineage={"dataset_rev": "f774f7a855ab", "mlflow_run_id": "train-run", "git_commit": "abc"},
)


def write_bundle(
    path: Path, manifest: Manifest = MANIFEST, extra: dict[str, bytes] | None = None
) -> str:
    """A seed bundle tarball plus its .sha256 sidecar; returns the digest."""
    files = {name: name.encode() for name in BUNDLE_FILES}
    files["manifest.json"] = json.dumps(
        {**manifest.__dict__, "classes": list(manifest.classes)}
    ).encode()
    files.update(extra or {})
    with tarfile.open(path, "w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    Path(f"{path}.sha256").write_text(f"{digest}  {path.name}\n")
    return digest


class FakeMlflow:
    """The slice of MlflowClient that seed() and current_champion() use."""

    def __init__(self, champion: str | None = None) -> None:
        self.versions: dict[str, dict[str, str]] = {}
        self.aliases: dict[str, str] = {}
        self.artifacts: dict[str, bytes] = {}
        self.run_status: str | None = None
        if champion is not None:
            self.versions[champion] = {
                "bundle_uri": f"s3://artifacts/models/cxr-classifier/{champion}/"
            }
            self.aliases["champion"] = champion

    def get_model_version_by_alias(self, name: str, alias: str) -> Any:
        if alias not in self.aliases:
            raise MlflowException("no alias", error_code=RESOURCE_DOES_NOT_EXIST)
        version = self.aliases[alias]
        return SimpleNamespace(version=version, tags=self.versions[version], run_id="run-x")

    def get_experiment_by_name(self, name: str) -> None:
        return None

    def create_experiment(self, name: str) -> str:
        return "1"

    def create_run(self, experiment_id: str, run_name: str, tags: dict[str, str]) -> Any:
        info = SimpleNamespace(
            run_id="seed-run", artifact_uri="mlflow-artifacts:/1/seed-run/artifacts"
        )
        return SimpleNamespace(info=info)

    def search_registered_models(self, filter_string: str) -> list[object]:
        return []

    def create_registered_model(self, name: str) -> None:
        pass

    def create_model_version(self, name: str, source: str, run_id: str, description: str) -> Any:
        version = str(len(self.versions) + 1)
        self.versions[version] = {}
        return SimpleNamespace(version=version)

    def log_artifacts(self, run_id: str, local_dir: str, artifact_path: str) -> None:
        for path in Path(local_dir).iterdir():
            self.artifacts[f"{artifact_path}/{path.name}"] = path.read_bytes()

    def set_model_version_tag(self, name: str, version: str, key: str, value: str) -> None:
        self.versions[version][key] = value

    def set_registered_model_alias(self, name: str, alias: str, version: str) -> None:
        self.aliases[alias] = version

    def set_terminated(self, run_id: str, status: str = "FINISHED") -> None:
        self.run_status = status


@pytest.fixture
def events(monkeypatch: pytest.MonkeyPatch) -> list[ChampionChanged]:
    sent: list[ChampionChanged] = []
    monkeypatch.setattr(seed_module, "publish_champion_changed", sent.append)
    return sent


def test_seed_registers_the_bundle_as_champion(
    tmp_path: Path, s3: FakeS3, events: list[ChampionChanged]
) -> None:
    archive = tmp_path / "cxr-seed-bundle-v2.tar.gz"
    digest = write_bundle(archive)
    mlflow = FakeMlflow()

    result = seed(str(archive), client=mlflow)  # type: ignore[arg-type]

    assert result.created
    assert result.model_version == "1"
    assert mlflow.aliases == {"champion": "1"}
    assert mlflow.run_status == "FINISHED"
    tags = mlflow.versions["1"]
    assert tags["bundle_uri"] == "s3://artifacts/models/cxr-classifier/1/"
    assert tags["dataset_rev"] == "f774f7a855ab"
    assert tags["origin_model_version"] == "2"
    assert tags["parity_passed"] == "true"
    assert json.loads(tags["thresholds"]) == {"A": 0.1, "B": 0.2}

    # The bundle that workers load says version 1 and keeps where it came from.
    stored = json.loads(s3.objects[("artifacts", "models/cxr-classifier/1/manifest.json")])
    assert stored["model_version"] == "1"
    assert stored["lineage"]["mlflow_run_id"] == "seed-run"
    assert stored["lineage"]["origin_mlflow_run_id"] == "train-run"
    assert stored["lineage"]["seed_bundle_sha256"] == digest
    assert ("artifacts", "models/cxr-classifier/1/model.onnx") in s3.objects
    assert json.loads(mlflow.artifacts["bundle/manifest.json"])["model_version"] == "1"
    assert events == [
        ChampionChanged("cxr-classifier", "1", "s3://artifacts/models/cxr-classifier/1/")
    ]


def test_seed_is_a_no_op_when_a_champion_exists(
    tmp_path: Path, s3: FakeS3, events: list[ChampionChanged]
) -> None:
    archive = tmp_path / "bundle.tar.gz"
    write_bundle(archive)
    mlflow = FakeMlflow(champion="4")

    result = seed(str(archive), client=mlflow)  # type: ignore[arg-type]

    assert not result.created
    assert result.model_version == "4"
    assert not s3.objects
    assert not events


def test_force_registers_over_an_existing_champion(
    tmp_path: Path, s3: FakeS3, events: list[ChampionChanged]
) -> None:
    archive = tmp_path / "bundle.tar.gz"
    write_bundle(archive)
    mlflow = FakeMlflow(champion="1")

    result = seed(str(archive), force=True, client=mlflow)  # type: ignore[arg-type]

    assert result.created
    assert mlflow.aliases["champion"] == result.model_version == "2"


def test_a_failed_registration_marks_the_run_failed(
    tmp_path: Path, s3: FakeS3, events: list[ChampionChanged], monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "bundle.tar.gz"
    write_bundle(archive)
    mlflow = FakeMlflow()

    def broken(*args: object) -> None:
        raise OSError("storage down")

    monkeypatch.setattr(seed_module.storage, "upload_file", broken)
    with pytest.raises(OSError):
        seed(str(archive), client=mlflow)  # type: ignore[arg-type]
    assert mlflow.run_status == "FAILED"
    assert "champion" not in mlflow.aliases
    assert not events


def test_fetch_rejects_a_digest_mismatch(tmp_path: Path) -> None:
    archive = tmp_path / "bundle.tar.gz"
    write_bundle(archive)
    with pytest.raises(SeedError, match="sha256 mismatch"):
        fetch(str(archive), tmp_path / "copy.tar.gz", expected_sha256="0" * 64)


def test_fetch_checks_the_sidecar_by_default(tmp_path: Path) -> None:
    archive = tmp_path / "bundle.tar.gz"
    write_bundle(archive)
    Path(f"{archive}.sha256").write_text("f" * 64 + "  bundle.tar.gz\n")
    with pytest.raises(SeedError, match="sha256 mismatch"):
        fetch(str(archive), tmp_path / "copy.tar.gz", expected_sha256=None)


def test_extract_writes_only_bundle_files(tmp_path: Path) -> None:
    archive = tmp_path / "bundle.tar.gz"
    write_bundle(archive, extra={"../evil.sh": b"rm -rf /"})
    out = tmp_path / "out"
    out.mkdir()

    assert extract(archive, out).model_version == "2"
    assert sorted(p.name for p in out.iterdir()) == sorted(BUNDLE_FILES)
    assert not (tmp_path / "evil.sh").exists()


def test_extract_rejects_a_bundle_for_other_preprocessing(tmp_path: Path) -> None:
    archive = tmp_path / "bundle.tar.gz"
    write_bundle(archive, manifest=Manifest(**{**MANIFEST.__dict__, "preprocess_version": "99"}))
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(SeedError, match="preprocess_version"):
        extract(archive, out)


def test_current_champion_is_none_before_seeding() -> None:
    assert current_champion(FakeMlflow()) is None  # type: ignore[arg-type]
    champion = current_champion(FakeMlflow(champion="3"))  # type: ignore[arg-type]
    assert champion is not None
    assert champion.model_version == "3"


@pytest.mark.django_db
def test_internal_champion_endpoint(internal: Client, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry_api, "current_champion", lambda: None)
    assert internal.get("/internal/champion").status_code == 404

    champion = Champion("cxr-classifier", "3", "s3://artifacts/models/cxr-classifier/3/", "r")
    monkeypatch.setattr(registry_api, "current_champion", lambda: champion)
    assert internal.get("/internal/champion").json() == {
        "model_name": "cxr-classifier",
        "model_version": "3",
        "bundle_uri": "s3://artifacts/models/cxr-classifier/3/",
        "mlflow_run_id": "r",
    }
