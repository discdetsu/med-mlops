"""`make demo`: the whole loop with no clicks, on the `full` stack.

    make up PROFILE=full && make demo
    uv run python scripts/demo.py [--n 200] [--feedback 0.6] [--min-new-labels N] [--seed S]

1. Traffic: AP views from NIH's train split that the serving model didn't train on, with NIH
   labels as reviewer feedback on some of them.
2. A monitor pass: drift is detected, and disagreements, the farthest cases, and a random sample
   become Label Studio tasks. Drift alone starts nothing.
3. Labeling: the real NIH labels stand in for a reviewer, through Label Studio's own API.
4. Export: labels → data/curated → new DVC rev → `should_retrain()` starts `train/retrain`.
5. Retrain in the flows-worker service → gate → `auto_promote` for this run only. A gate that
   fails ends the demo here, with its report: that is the gate doing its job.
6. Check: `@champion` moved, and a new upload is predicted by the new version on the new rev.

Each step leaves evidence, which the last lines point to. A CPU retrain takes about 30 minutes
with the full NIH dataset pinned. Run it again for another round: files already labeled are
skipped, so it picks others.
"""

import argparse
import asyncio
import random
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
import replay_traffic
from mlflow import MlflowClient
from prefect.flow_runs import wait_for_flow_run
from prefect.states import StateType

from cxr_core.registry import CHAMPION_ALIAS, MODEL_NAME, STATUS_APPROVED
from cxr_flows import db
from cxr_flows.config import load_config, load_labeling_config
from cxr_flows.dataset import sample_train
from cxr_flows.label_export import label_export_flow
from cxr_flows.labelstudio import NONE_CHOICE, LabelStudio, ensure_review_project
from cxr_flows.monitor import monitor_flow
from cxr_flows.registry import alias_version, newest_version
from cxr_flows.schema import CLASSES, read_metadata
from cxr_flows.settings import Paths, app_database_url, require_mlflow_tracking_uri

GOLDEN_IMAGE = "golden/png/00000001_000.png"


def step(text: str) -> None:
    print(f"\n=== {text}", flush=True)


def fail(text: str) -> None:
    sys.exit(f"\ndemo stopped: {text}")


def send_traffic(paths: Paths, args: argparse.Namespace) -> None:
    """Images the champion hasn't seen, so what comes back as labels is new to the next model."""
    train_rows = read_metadata(paths.nih)
    train_rows = train_rows[train_rows["split"] == "train"]
    cfg = load_config(load_labeling_config().train_config)
    seen = sample_train(train_rows, cfg.max_train_images, cfg.seed)["image_id"]
    with tempfile.NamedTemporaryFile("w", suffix=".txt") as exclude:
        exclude.write("\n".join(seen))
        exclude.flush()
        replay_traffic.main(
            [
                "--split", "train",
                "--slice", "view_position=AP",
                "--exclude-file", exclude.name,
                "--n", str(args.n),
                "--feedback", str(args.feedback),
                "--seed", str(args.seed),
                "--base-url", args.base_url,
            ]
        )  # fmt: skip


def label_with_nih(paths: Paths, url: str) -> int:
    """Answer every open task whose upload is an NIH image, as a reviewer would in the UI."""
    ls = LabelStudio.from_env()
    if ls is None:
        fail("LABEL_STUDIO_URL is not set; run through `make demo`")
        raise AssertionError  # fail() exits
    project = ensure_review_project(ls, load_labeling_config().project_title)
    nih = read_metadata(paths.nih).set_index("image_id")
    unanswered = {int(t["id"]) for t in ls.list_tasks(project) if not t.get("annotations")}
    answered = 0
    for task in db.read_tasks(url):
        image_id = Path(task.filename).stem
        if task.ls_task_id not in unanswered or image_id not in nih.index:
            continue
        row = nih.loc[image_id]
        ls.annotate(task.ls_task_id, [c for c in CLASSES if row[c] == 1] or [NONE_CHOICE])
        answered += 1
    return answered


def predict_once(paths: Paths, base_url: str, want_version: str) -> dict[str, Any]:
    """Upload a golden image until a worker on `want_version` answers; the swap takes a moment."""
    path = paths.golden / Path(GOLDEN_IMAGE).relative_to("golden")
    deadline = time.monotonic() + 120
    with httpx.Client(base_url=base_url, timeout=30) as client:
        while time.monotonic() < deadline:
            with path.open("rb") as f:
                job = client.post("/api/jobs", files={"file": (path.name, f, "image/png")}).json()
            for _ in range(60):
                body = client.get(f"/api/jobs/{job['id']}").json()
                if body["status"] != "queued":
                    break
                time.sleep(1)
            prediction = body.get("prediction") or {}
            if prediction.get("model_version") == want_version:
                result: dict[str, Any] = prediction
                return result
            time.sleep(2)
    fail(f"no prediction from version {want_version} within 2 minutes; check the workers' logs")
    raise AssertionError  # fail() exits


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--n", type=int, default=200, help="images to send (default 200)")
    parser.add_argument("--feedback", type=float, default=0.6, help="share given feedback")
    parser.add_argument("--min-new-labels", type=int, help="override N from labeling.yaml")
    parser.add_argument("--seed", type=int, default=random.randrange(2**31))
    parser.add_argument("--timeout", type=int, default=5400, help="seconds to wait for the retrain")
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()

    paths = Paths.discover()
    url = app_database_url()
    client = MlflowClient(require_mlflow_tracking_uri())
    champion = alias_version(client, MODEL_NAME, CHAMPION_ALIAS)
    if champion is None:
        fail("no @champion to start from; run `make seed-model`")
    print(f"@champion is version {champion}; seed {args.seed}")

    step(f"1/6 Traffic: {args.n} AP views the champion didn't train on")
    send_traffic(paths, args)

    step("2/6 Monitor pass: drift → cases for review")
    reports = monitor_flow()
    current = next((r for r in reports if r.is_champion), None)
    if current is None or not current.drift_detected:
        fail("the monitor pass detected no drift on @champion. Run it again, or raise --n.")
    print(f"tasks so far: {db.task_counts(url)}")

    # One lock across both steps: a scheduled export in between would take these answers and start
    # the retrain without auto_promote.
    with db.labeling_lock(url):
        step("3/6 Labeling: NIH labels stand in for a reviewer")
        print(f"{label_with_nih(paths, url)} tasks answered in Label Studio")

        step("4/6 Export: labels → data/curated → new DVC rev → should_retrain()")
        result = label_export_flow(
            auto_promote=True, min_new_labels=args.min_new_labels, lock_held=True
        )
    decision = result.decision
    print(f"dataset_rev {result.dataset_rev}: {result.exported} exported, "
          f"{result.quarantined} quarantined")  # fmt: skip
    if decision is None or not result.train_flow_run_id:
        reason = decision.reason if decision else "nothing exported, or a retrain is still running"
        fail(f"no retrain: {reason}. Run it again for more labels, or lower --min-new-labels.")
    print(f"retrain started: {decision.reason}")

    step("5/6 Retrain → gate → auto_promote (flows-worker; about 30 minutes on a laptop CPU)")
    print(f"follow it at http://localhost:4200/runs/flow-run/{result.train_flow_run_id}")
    run = asyncio.run(wait_for_flow_run(uuid.UUID(result.train_flow_run_id), timeout=args.timeout))
    state = run.state
    if state is None or state.type != StateType.COMPLETED:
        fail(f"the retrain ended {state.name if state else 'in an unknown state'}: see the link")
    version = newest_version(client, MODEL_NAME)
    if version is None or version.tags.get("prefect_flow_run_id") != result.train_flow_run_id:
        fail("the retrain registered no model version")
    candidate = db.candidate_status(url, MODEL_NAME, version.version)
    print(f"version {version.version} on dataset_rev {version.tags.get('dataset_rev')}")
    if candidate is None or not candidate["gate_passed"]:
        report = candidate["report_uri"] if candidate else "none"
        fail(f"version {version.version} didn't pass the gate, so it isn't promoted: {report}")
    print(f"gate passed: {candidate['report_uri']}")
    if candidate["status"] != STATUS_APPROVED:
        fail(f"version {version.version} passed but wasn't promoted: {candidate['status']}")

    step("6/6 Check: the new version serves")
    now = alias_version(client, MODEL_NAME, CHAMPION_ALIAS)
    if now != version.version:
        fail(f"@champion is {now}, expected {version.version}")
    prediction = predict_once(paths, args.base_url, version.version)
    if prediction["dataset_rev"] != result.dataset_rev:
        fail(f"prediction names dataset_rev {prediction['dataset_rev']}, not {result.dataset_rev}")
    print(f"@champion moved {champion} → {now}; a new upload was predicted by version "
          f"{prediction['model_version']} on dataset_rev {prediction['dataset_rev']}")  # fmt: skip
    label_export_flow(trigger=False)  # refresh the labeling gauges for Grafana
    monitor_flow()

    print(f"""
The loop closed. Evidence:
  Label Studio  http://localhost:8080   project "{load_labeling_config().project_title}": the answered tasks
  MLflow        http://localhost:5000   version {now}, tagged dataset_rev {result.dataset_rev}
  Django admin  http://localhost:8000/admin/  Labeling (tasks, retrain decisions), Registry (the
                candidate, gate report, approval by auto_promote), Monitoring (drift reports)
  Grafana       http://localhost:3000   CXR · Model: drift, labeling tasks, new labels
  Prefect       http://localhost:4200   the monitor, label-export, train, and gate runs
Commit the data pointer: git add data/curated.dvc && git commit""")  # noqa: E501


if __name__ == "__main__":
    main()
