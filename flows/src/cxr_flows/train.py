"""Prefect `train` flow: validate → build dataset (pinned DVC rev) → train → evaluate →
export ONNX → E2E parity → build reference → register as `@challenger`.

    uv run python -m cxr_flows.train [--config demo|full|path.yaml] [--dataset-rev REV]

Without PREFECT_API_URL, Prefect 3 runs the flow against a temporary local server, so this
works under the `core` profile; with it set (`make train PROFILE=full`), runs show up in the
Prefect UI.
"""

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import torch
from prefect import flow, get_run_logger, runtime, task
from prefect.cache_policies import NO_CACHE

from cxr_core.bundle import REFERENCE_FILE, Manifest, save_head
from cxr_core.preprocess import PREPROCESS_VERSION
from cxr_flows import golden as golden_set
from cxr_flows.config import TrainConfig, load_config
from cxr_flows.dataset import Split, load_split
from cxr_flows.evaluate import choose_thresholds, evaluate, render_html
from cxr_flows.export import ParityReport, export_onnx, run_parity
from cxr_flows.reference import build_reference, write_reference
from cxr_flows.registry import register
from cxr_flows.schema import CLASSES, check_no_shared_patients, read_metadata, validate_dataset
from cxr_flows.settings import MODEL_NAME, TRAIN_EXPERIMENT, Paths
from cxr_flows.training import TrainResult, predict, train_model
from cxr_flows.versioning import (
    DatasetChangedError,
    dataset_rev,
    dvc_md5,
    dvc_pull,
    git_commit,
    is_clean,
)


class ParityError(RuntimeError):
    """ONNX and PyTorch disagree on the golden set; the model is not registered."""


@dataclass(frozen=True)
class PinnedData:
    rev: str
    md5s: dict[str, str]
    nih: pd.DataFrame
    test_locked: pd.DataFrame
    golden: list[tuple[Path, str]]


@dataclass(frozen=True)
class Splits:
    train: Split
    val: Split
    test: Split


@task(cache_policy=NO_CACHE)
def validate_data(paths: Paths, expected_rev: str | None) -> PinnedData:
    """Pull missing data, refuse a workspace that differs from its .dvc pins, then validate."""
    for target in (paths.nih, paths.test_locked, paths.golden):
        if not target.exists():
            dvc_pull(paths.root, [target])
        if not is_clean(paths.root, target):
            rel = target.relative_to(paths.root)
            raise DatasetChangedError(
                f"{rel} differs from {rel}.dvc; run `dvc checkout {rel}` or `make seed`"
            )

    md5s = {str(t.relative_to(paths.root)): dvc_md5(t) for t in (paths.nih, paths.test_locked)}
    rev = dataset_rev(md5s)
    if expected_rev is not None and rev != expected_rev:
        raise DatasetChangedError(
            f"workspace data is dataset_rev {rev}, expected {expected_rev}; check out the git "
            "commit whose data/*.dvc files have that rev, then `dvc checkout`"
        )

    nih = validate_dataset(read_metadata(paths.nih), ("train", "val"), paths.nih)
    locked = validate_dataset(read_metadata(paths.test_locked), ("test_locked",), paths.test_locked)
    check_no_shared_patients({"data/nih": nih, "data/test_locked": locked})
    manifest = golden_set.read_manifest(paths.golden)
    golden = [
        (paths.golden / f, i) for f, i in zip(manifest["file"], manifest["image_id"], strict=True)
    ]
    get_run_logger().info("dataset_rev %s: %s", rev, md5s)
    return PinnedData(rev, md5s, nih, locked, golden)


@task(cache_policy=NO_CACHE)
def build_splits(paths: Paths, data: PinnedData, cfg: TrainConfig) -> Splits:
    cache = paths.cache
    nih_md5, locked_md5 = data.md5s["data/nih"], data.md5s["data/test_locked"]
    nih = data.nih
    train = load_split(
        nih[nih["split"] == "train"], paths.nih, nih_md5, "train", cfg.image_size, cache
    )
    if cfg.max_train_images is not None and cfg.max_train_images < len(train):
        rng = np.random.default_rng(cfg.seed)
        keep = np.sort(rng.choice(len(train), cfg.max_train_images, replace=False))
        train = Split(train.meta.iloc[keep].reset_index(drop=True), train.x[keep], train.y[keep])
    val = load_split(nih[nih["split"] == "val"], paths.nih, nih_md5, "val", cfg.image_size, cache)
    test = load_split(
        data.test_locked, paths.test_locked, locked_md5, "test_locked", cfg.image_size, cache
    )
    get_run_logger().info("train %d, val %d, test_locked %d", len(train), len(val), len(test))
    return Splits(train, val, test)


@task(cache_policy=NO_CACHE)
def train(cfg: TrainConfig, splits: Splits) -> TrainResult:
    return train_model(cfg, splits.train, splits.val, len(CLASSES))


@task(cache_policy=NO_CACHE)
def evaluate_model(
    cfg: TrainConfig, result: TrainResult, splits: Splits, out: Path
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    """Thresholds from val; reports on test_locked (and the recent slice, from Phase 5)."""
    batch = cfg.batch_size * 2
    val_probs = predict(result.model, splits.val.x, batch).probabilities
    thresholds = choose_thresholds(splits.val.y, val_probs, CLASSES)
    test_probs = predict(result.model, splits.test.x, batch).probabilities

    def report(split: Split, probs: np.ndarray) -> dict[str, Any]:
        return evaluate(
            split.y,
            probs,
            split.meta,
            CLASSES,
            thresholds,
            target_specificity=cfg.target_specificity,
            bootstrap_samples=cfg.bootstrap_samples,
            min_subgroup_size=cfg.min_subgroup_size,
            seed=cfg.seed,
        )

    reports = {"test_locked": report(splits.test, test_probs), "val": report(splits.val, val_probs)}
    # The recent slice (held-back reviewer labels) is evaluated here too once Phase 5 exports
    # one; until then the report says so explicitly.
    payload = {**reports, "recent": None}
    (out / "evaluation.json").write_text(json.dumps(payload, indent=2) + "\n")
    (out / "evaluation.html").write_text(render_html("CXR evaluation", reports))
    return thresholds, reports


@task(cache_policy=NO_CACHE)
def export(cfg: TrainConfig, result: TrainResult, bundle: Path) -> Path:
    return export_onnx(result.model, cfg.image_size, bundle / "model.onnx")


@task(cache_policy=NO_CACHE)
def parity(
    cfg: TrainConfig, result: TrainResult, onnx_path: Path, data: PinnedData
) -> ParityReport:
    return run_parity(result.model, onnx_path, data.golden, cfg.image_size, cfg.parity)


@task(cache_policy=NO_CACHE)
def reference(cfg: TrainConfig, result: TrainResult, splits: Splits, bundle: Path) -> Path:
    preds = predict(result.model, splits.val.x, cfg.batch_size * 2)
    return write_reference(
        build_reference(splits.val.meta, preds, CLASSES), bundle / REFERENCE_FILE
    )


@flow(name="train", log_prints=True)
def train_flow(
    config: str = "demo", dataset_rev: str | None = None, register_model: bool = True
) -> str | None:
    """Returns the registered model version, or None with `register_model=False`."""
    cfg = load_config(config)
    paths = Paths.discover()
    logger = get_run_logger()

    data = validate_data(paths, dataset_rev)
    splits = build_splits(paths, data, cfg)
    commit = git_commit(paths.root)
    flow_run_id = str(runtime.flow_run.id or "")
    lineage = {
        "dataset_rev": data.rev,
        "git_commit": commit,
        "preprocess_version": PREPROCESS_VERSION,
        "prefect_flow_run_id": flow_run_id,
        "config_hash": cfg.config_hash,
        "profile": cfg.profile,
        **{f"dvc_md5.{path}": md5 for path, md5 in data.md5s.items()},
    }

    mlflow.set_experiment(TRAIN_EXPERIMENT)
    with (
        mlflow.start_run(run_name=f"{cfg.profile}-{data.rev}") as run,
        tempfile.TemporaryDirectory() as tmp,
    ):
        mlflow.set_tags(lineage)
        sizes = {"n_train": len(splits.train), "n_val": len(splits.val), "n_test": len(splits.test)}
        mlflow.log_params({**cfg.model_dump(exclude={"parity"}), **sizes})
        mlflow.log_dict(cfg.model_dump(), "config.json")

        result = train(cfg, splits)
        for row in result.history:
            step = int(row["epoch"])
            mlflow.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=step)
        mlflow.log_metric("best_epoch", result.best_epoch)

        out, bundle = Path(tmp) / "reports", Path(tmp) / "bundle"
        out.mkdir()
        bundle.mkdir()
        thresholds, reports = evaluate_model(cfg, result, splits, out)
        test_report = reports["test_locked"]
        mlflow.log_metric("test_macro_auroc", test_report["macro_auroc"] or float("nan"))
        mlflow.log_metric("test_ece", test_report["ece"] or float("nan"))
        for c, m in test_report["classes"].items():
            mlflow.log_metric(f"test_auroc.{c}", m["auroc"] or float("nan"))

        onnx_path = export(cfg, result, bundle)
        parity_report = parity(cfg, result, onnx_path, data)
        (out / "parity.json").write_text(json.dumps(parity_report.to_dict(), indent=2) + "\n")
        mlflow.log_artifacts(str(out), artifact_path="reports")
        mlflow.set_tag("parity_passed", str(parity_report.passed).lower())
        if not parity_report.passed:
            raise ParityError("E2E parity failed: " + "; ".join(parity_report.failures))
        logger.info("parity passed on %d golden files", len(parity_report.files))

        reference(cfg, result, splits, bundle)
        head = result.model.head
        save_head(bundle, head.weight.detach().numpy(), head.bias.detach().numpy())
        # PyTorch weights too, for fine-tuning from this version later; serving never uses them.
        torch.save(result.model.state_dict(), Path(tmp) / "model.pt")
        mlflow.log_artifact(str(Path(tmp) / "model.pt"), artifact_path="pytorch")

        if not register_model:
            logger.info("register_model=False: run %s logged, nothing registered", run.info.run_id)
            return None

        draft = Manifest(
            model_name=MODEL_NAME,
            model_version="pending",
            architecture=cfg.architecture,
            classes=CLASSES,
            image_size=cfg.image_size,
            preprocess_version=PREPROCESS_VERSION,
            thresholds=thresholds,
            lineage={**lineage, "mlflow_run_id": run.info.run_id},
        )
        version = register(
            run.info.run_id,
            bundle,
            draft,
            {
                **lineage,
                "parity_passed": "true",
                "architecture": cfg.architecture,
                "image_size": str(cfg.image_size),
                "classes": json.dumps(list(CLASSES)),
                "test_macro_auroc": f"{test_report['macro_auroc']:.4f}",
            },
        )
    logger.info("registered %s version %s as @challenger", MODEL_NAME, version)
    return version


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default="demo", help="profile name or YAML path")
    parser.add_argument("--dataset-rev", help="fail unless the workspace data has this rev")
    parser.add_argument("--no-register", action="store_true", help="log the run only")
    args = parser.parse_args()
    train_flow(
        config=args.config, dataset_rev=args.dataset_rev, register_model=not args.no_register
    )


if __name__ == "__main__":
    main()
