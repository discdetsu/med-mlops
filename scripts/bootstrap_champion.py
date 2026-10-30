"""Make the first registered model `@champion` and export the seed bundle for `core`.

    uv run python scripts/bootstrap_champion.py [--version N] [--force] [--out dist]

Without --version it takes the current `@challenger`. It refuses when a `@champion` already
exists, because from then on promotion goes through the gate and Django admin (Phase 3);
--force overrides that for local resets. The version must have passed E2E parity.

The seed bundle is the version's serving bundle (ONNX, head, reference, thresholds in the
manifest) as dist/cxr-seed-bundle-v<N>.tar.gz plus a .sha256, ready to attach to a GitHub
Release that `make seed-model` downloads:

    gh release create seed-v<N> dist/cxr-seed-bundle-v<N>.tar.gz{,.sha256}

then point SEED_BUNDLE in the Makefile at it. `make seed-model SEED_BUNDLE=dist/...` registers
a local bundle instead.

Check the NIH ChestX-ray14 terms before publishing weights trained on it.
"""

import argparse
import hashlib
import sys
import tarfile
import tempfile
from pathlib import Path

import boto3
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from cxr_core.bundle import BUNDLE_FILES, Manifest
from cxr_core.registry import (
    ARTIFACTS_BUCKET,
    CHALLENGER_ALIAS,
    CHAMPION_ALIAS,
    MODEL_NAME,
    bundle_prefix,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--version", help=f"model version; default: @{CHALLENGER_ALIAS}")
    parser.add_argument("--force", action="store_true", help="replace an existing @champion")
    parser.add_argument("--out", type=Path, default=Path("dist"))
    args = parser.parse_args()

    client = MlflowClient()
    champion = _alias_version(client, CHAMPION_ALIAS)
    if champion is not None and not args.force:
        print(
            f"@{CHAMPION_ALIAS} is already version {champion}; promote through the gate, "
            "or pass --force to reset it",
            file=sys.stderr,
        )
        return 1

    version = args.version or _alias_version(client, CHALLENGER_ALIAS)
    if version is None:
        print(f"no --version and no @{CHALLENGER_ALIAS}; run `make train` first", file=sys.stderr)
        return 1
    mv = client.get_model_version(MODEL_NAME, version)
    if mv.tags.get("parity_passed") != "true":
        print(f"version {version} has not passed E2E parity", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    bundle = args.out / f"cxr-seed-bundle-v{version}.tar.gz"
    with tempfile.TemporaryDirectory() as tmp:
        bundle_dir = Path(tmp)
        s3 = boto3.client("s3")
        prefix = bundle_prefix(MODEL_NAME, version)
        for name in BUNDLE_FILES:
            s3.download_file(ARTIFACTS_BUCKET, prefix + name, str(bundle_dir / name))
        manifest = Manifest.load(bundle_dir)  # fails on a malformed or mismatched bundle
        if manifest.model_version != version:
            print(f"bundle manifest says version {manifest.model_version}", file=sys.stderr)
            return 1
        with tarfile.open(bundle, "w:gz") as tar:
            for name in BUNDLE_FILES:
                tar.add(bundle_dir / name, arcname=name)

    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    bundle.with_name(bundle.name + ".sha256").write_text(f"{digest}  {bundle.name}\n")

    client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, version)
    print(f"{MODEL_NAME} version {version} is now @{CHAMPION_ALIAS}")
    print(f"seed bundle: {bundle} (sha256 {digest[:12]}…)")
    print(f"publish: gh release create seed-v{version} {bundle} {bundle}.sha256")
    return 0


def _alias_version(client: MlflowClient, alias: str) -> str | None:
    try:
        return str(client.get_model_version_by_alias(MODEL_NAME, alias).version)
    except MlflowException:
        return None


if __name__ == "__main__":
    sys.exit(main())
