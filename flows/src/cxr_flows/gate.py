"""Prefect `gate` flow: gate `@challenger` (or a given version) against `@champion` on the
locked test set, publish the gate report, and record the version as a candidate that a person
approves or rejects in Django admin (ADR 0007).

    uv run python -m cxr_flows.gate [--version N] [--config gate|path.yaml]

The train flow runs it after registering a version. Run it by hand (`make gate`) to re-gate a
candidate after `@champion` moved: approval refuses a candidate gated against another champion.

Both versions are scored the way serving scores them: `cxr_core.preprocess` from the raw
files (cached tensors) → each version's own ONNX model through `cxr_core.inference`.
"""

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import boto3
import numpy as np
import onnxruntime as ort
import pandas as pd
from mlflow import MlflowClient
from prefect import flow, get_run_logger, runtime, task
from prefect.cache_policies import NO_CACHE

from cxr_core.bundle import MANIFEST_FILE, ONNX_FILE, Manifest
from cxr_core.inference import run_onnx, sigmoid
from cxr_core.preprocess import PREPROCESS_VERSION
from cxr_core.registry import (
    CHALLENGER_ALIAS,
    CHAMPION_ALIAS,
    MODEL_NAME,
    bundle_uri,
    gate_report_prefix,
    split_s3_uri,
)
from cxr_flows import db
from cxr_flows.config import load_gate_config
from cxr_flows.dataset import Split, load_split
from cxr_flows.gating import Scored, decide, render_html, version_tags
from cxr_flows.registry import alias_version
from cxr_flows.schema import CLASSES, read_metadata, validate_dataset
from cxr_flows.settings import Paths, app_database_url, require_mlflow_tracking_uri
from cxr_flows.versioning import dvc_md5, ensure_pinned

ADMIN_CANDIDATES_URL = "http://localhost:8000/admin/registry/modelcandidate/"


class GateError(RuntimeError):
    """The gate can't compare the two versions (as opposed to a comparison that fails)."""


@dataclass(frozen=True)
class LockedTest:
    meta: pd.DataFrame
    md5: str


@dataclass(frozen=True)
class GateOutcome:
    model_version: str
    champion_version: str
    passed: bool
    report_uri: str


@task(cache_policy=NO_CACHE)
def load_locked_test(paths: Paths) -> LockedTest:
    """The pinned test_locked metadata. `ensure_pinned` already proves the files are the
    pinned ones, so the schema check skips opening 16k images."""
    ensure_pinned(paths.root, [paths.test_locked])
    meta = validate_dataset(read_metadata(paths.test_locked), ("test_locked",))
    return LockedTest(meta, dvc_md5(paths.test_locked))


@task(cache_policy=NO_CACHE)
def score_version(
    client: MlflowClient, version: str, test: LockedTest, paths: Paths, batch_size: int, work: Path
) -> tuple[Scored, Split]:
    """Download the version's bundle and score the locked test set with its ONNX model."""
    mv = client.get_model_version(MODEL_NAME, version)
    bucket, prefix = split_s3_uri(mv.tags.get("bundle_uri") or bundle_uri(MODEL_NAME, version))
    dest = work / f"v{version}"
    dest.mkdir()
    s3 = boto3.client("s3")
    for name in (MANIFEST_FILE, ONNX_FILE):
        s3.download_file(bucket, prefix + name, str(dest / name))
    manifest = Manifest.load(dest)
    if manifest.classes != CLASSES:
        raise GateError(f"version {version} predicts {manifest.classes}, the gate needs {CLASSES}")
    if manifest.preprocess_version != PREPROCESS_VERSION:
        raise GateError(
            f"version {version} needs preprocess_version {manifest.preprocess_version}, this code "
            f"is {PREPROCESS_VERSION}; serving would refuse it too"
        )

    split = load_split(
        test.meta, paths.test_locked, test.md5, "test_locked", manifest.image_size, paths.cache
    )
    session = ort.InferenceSession(str(dest / ONNX_FILE), providers=["CPUExecutionProvider"])
    probs = np.concatenate(
        [
            sigmoid(run_onnx(session, np.ascontiguousarray(split.x[i : i + batch_size]))[0])
            for i in range(0, len(split), batch_size)
        ]
    )
    get_run_logger().info(
        "version %s (%s, %d px) scored %d images", version, manifest.architecture,
        manifest.image_size, len(split),
    )  # fmt: skip
    return Scored(version, probs, parity_passed=mv.tags.get("parity_passed") == "true"), split


@task(cache_policy=NO_CACHE)
def publish_report(
    client: MlflowClient, report: dict[str, Any], run_id: str | None, work: Path
) -> str:
    """JSON + HTML next to the version's bundle in S3, and in its MLflow run under `gate/`.
    Returns the S3 URI of the HTML report, which Django admin links to."""
    challenger, champion = report["challenger_version"], report["champion_version"]
    out = work / "gate"
    out.mkdir()
    stem = f"vs-champion-v{champion}"
    files = {
        out / f"{stem}.json": ("application/json", json.dumps(report, indent=2) + "\n"),
        out / f"{stem}.html": ("text/html; charset=utf-8", render_html(report)),
    }
    s3 = boto3.client("s3")
    bucket, _ = split_s3_uri(bundle_uri(MODEL_NAME, challenger))
    key_prefix = gate_report_prefix(MODEL_NAME, challenger)
    for path, (content_type, text) in files.items():
        path.write_text(text)
        s3.upload_file(str(path), bucket, key_prefix + path.name, {"ContentType": content_type})
        if run_id:
            client.log_artifact(run_id, str(path), artifact_path="gate")
    return f"s3://{bucket}/{key_prefix}{stem}.html"


@flow(name="gate", log_prints=True)
def gate_flow(version: str | None = None, config: str = "gate") -> GateOutcome | None:
    """Gate `version` (default: `@challenger`). None when there is nothing to compare."""
    cfg = load_gate_config(config)
    logger = get_run_logger()
    client = MlflowClient(require_mlflow_tracking_uri())

    champion = alias_version(client, MODEL_NAME, CHAMPION_ALIAS)
    challenger = version or alias_version(client, MODEL_NAME, CHALLENGER_ALIAS)
    if challenger is None:
        raise GateError(f"no version given and no @{CHALLENGER_ALIAS}; run `make train` first")
    if champion is None:
        logger.warning(
            "no @%s to compare version %s with; `make seed-model` or `make bootstrap-champion` "
            "sets the first one", CHAMPION_ALIAS, challenger,
        )  # fmt: skip
        return None
    if champion == challenger:
        logger.info("version %s is already @%s; nothing to gate", challenger, CHAMPION_ALIAS)
        return None

    mv = client.get_model_version(MODEL_NAME, challenger)
    paths = Paths.discover()
    test = load_locked_test(paths)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        champion_scores, _ = score_version(client, champion, test, paths, cfg.batch_size, work)
        challenger_scores, split = score_version(
            client, challenger, test, paths, cfg.batch_size, work
        )
        report = decide(cfg, split.y, split.meta, CLASSES, champion_scores, challenger_scores)
        report |= {
            "model_name": MODEL_NAME,
            "test_locked_md5": test.md5,
            "prefect_flow_run_id": str(runtime.flow_run.id or ""),
        }
        report_uri = publish_report(client, report, mv.run_id, work)

    passed = bool(report["passed"])
    for key, value in version_tags(report, report_uri).items():
        client.set_model_version_tag(MODEL_NAME, challenger, key, value)
    db.record_candidate(
        app_database_url(),
        model_name=MODEL_NAME,
        model_version=challenger,
        champion_version=champion,
        gate_passed=passed,
        auto_promote=cfg.auto_promote and passed,
        report=report,
        report_uri=report_uri,
        mlflow_run_id=mv.run_id or "",
        dataset_rev=mv.tags.get("dataset_rev", ""),
    )

    for check in report["checks"]:
        logger.info("%s %s: %s", "pass" if check["passed"] else "FAIL", check["name"],
                    check["detail"])  # fmt: skip
    if not passed:
        logger.warning("version %s FAILED the gate against version %s", challenger, champion)
    elif cfg.auto_promote:
        logger.info("version %s passed; auto_promote is on, `promote --auto` takes it", challenger)
    else:
        logger.info("version %s passed; approve or reject it at %s", challenger,
                    ADMIN_CANDIDATES_URL)  # fmt: skip
    return GateOutcome(challenger, champion, passed, report_uri)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--version", help=f"model version to gate; default: @{CHALLENGER_ALIAS}")
    parser.add_argument("--config", default="gate", help="gate config name or YAML path")
    args = parser.parse_args()
    outcome = gate_flow(version=args.version, config=args.config)
    if outcome is not None:
        verdict = "passed" if outcome.passed else "FAILED"
        print(f"version {outcome.model_version} {verdict} the gate against version "
              f"{outcome.champion_version}: {outcome.report_uri}")  # fmt: skip


if __name__ == "__main__":
    main()
