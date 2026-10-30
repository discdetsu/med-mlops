"""Register a seed bundle (from `scripts/bootstrap_champion.py`) as `@champion`, so `core`
serves predictions on a fresh clone without training anything.

The bundle becomes a new version in this machine's MLflow. Its manifest's `model_version`
is rewritten to that version; the version it had where it was trained is kept in the
lineage as `origin_model_version`, next to the training run (`origin_mlflow_run_id`).
"""

import dataclasses
import hashlib
import json
import shutil
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from mlflow import MlflowClient

from cxr_backend import storage
from cxr_backend.registry.champion import current_champion, mlflow_client
from cxr_backend.registry.events import publish_champion_changed
from cxr_core.bundle import BUNDLE_FILES, Manifest
from cxr_core.messaging import ChampionChanged
from cxr_core.preprocess import PREPROCESS_VERSION
from cxr_core.registry import (
    ARTIFACTS_BUCKET,
    CHAMPION_ALIAS,
    MODEL_NAME,
    bundle_prefix,
    bundle_uri,
)

SEED_EXPERIMENT = "cxr-seed"
_DOWNLOAD_TIMEOUT_S = 60


class SeedError(Exception):
    pass


@dataclass(frozen=True)
class SeedResult:
    model_version: str
    created: bool  # False when a champion already existed and nothing was done


def fetch(source: str, dest: Path, expected_sha256: str | None) -> str:
    """Copy a local path or download an http(s) URL to `dest`; return its sha256.

    For URLs without `expected_sha256`, `<url>.sha256` (written by bootstrap_champion.py)
    must exist; a download is never registered unverified.
    """
    is_url = source.startswith(("https://", "http://"))
    if is_url:
        with (
            urllib.request.urlopen(source, timeout=_DOWNLOAD_TIMEOUT_S) as resp,
            dest.open("wb") as f,
        ):
            shutil.copyfileobj(resp, f)
        if expected_sha256 is None:
            with urllib.request.urlopen(source + ".sha256", timeout=_DOWNLOAD_TIMEOUT_S) as resp:
                expected_sha256 = resp.read().decode().split()[0]
    else:
        shutil.copyfile(source, dest)
        sidecar = Path(source + ".sha256")
        if expected_sha256 is None and sidecar.exists():
            expected_sha256 = sidecar.read_text().split()[0]

    digest = hashlib.sha256()
    with dest.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if expected_sha256 is not None and actual != expected_sha256.lower():
        raise SeedError(f"sha256 mismatch: expected {expected_sha256}, got {actual}")
    return actual


def extract(archive: Path, dest: Path) -> Manifest:
    """Write exactly the bundle files from `archive` into `dest`, nothing else.

    Members are read by name and written by this code, so paths in the archive never reach
    the filesystem.
    """
    with tarfile.open(archive, "r:gz") as tar:
        for name in BUNDLE_FILES:
            try:
                member = tar.getmember(name)
            except KeyError:
                raise SeedError(f"{archive.name} has no {name}") from None
            src = tar.extractfile(member) if member.isreg() else None
            if src is None:
                raise SeedError(f"{archive.name}: {name} is not a regular file")
            with src, (dest / name).open("wb") as out:
                shutil.copyfileobj(src, out)
    manifest = Manifest.load(dest)
    if manifest.model_name != MODEL_NAME:
        raise SeedError(f"bundle is for {manifest.model_name!r}, this stack serves {MODEL_NAME!r}")
    if manifest.preprocess_version != PREPROCESS_VERSION:
        raise SeedError(
            f"bundle needs preprocess_version {manifest.preprocess_version}, "
            f"this code is {PREPROCESS_VERSION}"
        )
    return manifest


def seed(
    source: str,
    expected_sha256: str | None = None,
    force: bool = False,
    client: MlflowClient | None = None,
) -> SeedResult:
    client = client or mlflow_client()
    existing = current_champion(client)
    if existing is not None and not force:
        return SeedResult(existing.model_version, created=False)

    with tempfile.TemporaryDirectory() as tmp:
        archive, bundle_dir = Path(tmp) / "bundle.tar.gz", Path(tmp) / "bundle"
        bundle_dir.mkdir()
        digest = fetch(source, archive, expected_sha256)
        manifest = extract(archive, bundle_dir)
        version = _register(client, manifest, bundle_dir, Path(source).name, digest)

    publish_champion_changed(ChampionChanged(MODEL_NAME, version, bundle_uri(MODEL_NAME, version)))
    return SeedResult(version, created=True)


def _register(
    client: MlflowClient, origin: Manifest, bundle_dir: Path, bundle_name: str, digest: str
) -> str:
    experiment = client.get_experiment_by_name(SEED_EXPERIMENT)
    experiment_id = (
        experiment.experiment_id if experiment else client.create_experiment(SEED_EXPERIMENT)
    )
    lineage = dict(origin.lineage)
    run = client.create_run(
        experiment_id,
        run_name=f"seed-{bundle_name}",
        tags={**lineage, "seed_bundle": bundle_name, "seed_bundle_sha256": digest},
    )
    run_id = run.info.run_id
    try:
        if not client.search_registered_models(filter_string=f"name = '{MODEL_NAME}'"):
            client.create_registered_model(MODEL_NAME)
        mv = client.create_model_version(
            MODEL_NAME,
            source=f"{run.info.artifact_uri}/bundle",
            run_id=run_id,
            description=f"Seeded from {bundle_name} (sha256 {digest[:12]})",
        )
        version = str(mv.version)

        lineage.update(
            mlflow_run_id=run_id,
            origin_model_version=origin.model_version,
            origin_mlflow_run_id=origin.lineage.get("mlflow_run_id", ""),
            seed_bundle_sha256=digest,
        )
        manifest = dataclasses.replace(origin, model_version=version, lineage=lineage)
        manifest.save(bundle_dir)
        client.log_artifacts(run_id, str(bundle_dir), artifact_path="bundle")
        prefix = bundle_prefix(MODEL_NAME, version)
        for name in BUNDLE_FILES:
            storage.upload_file(str(bundle_dir / name), ARTIFACTS_BUCKET, prefix + name)

        tags = {
            **lineage,
            # Only versions that passed E2E parity are bundled (bootstrap_champion.py checks).
            "parity_passed": "true",
            "architecture": manifest.architecture,
            "image_size": str(manifest.image_size),
            "classes": json.dumps(list(manifest.classes)),
            "thresholds": json.dumps(manifest.thresholds, sort_keys=True),
            "bundle_uri": bundle_uri(MODEL_NAME, version),
            "seed_bundle": bundle_name,
        }
        for key, value in tags.items():
            client.set_model_version_tag(MODEL_NAME, version, key, value)
        client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, version)
    except BaseException:
        client.set_terminated(run_id, status="FAILED")
        raise
    client.set_terminated(run_id)
    return version
