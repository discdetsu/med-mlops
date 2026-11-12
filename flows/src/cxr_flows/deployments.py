"""The `flows-worker` service: serves the deployments that need the workspace and torch.

    python -m cxr_flows.deployments

- `train/retrain`, started by `should_retrain()`, never on a schedule;
- `label-export/scheduled`, every `export_interval_seconds`.

One run at a time (`limit=1`): both read and write the DVC workspace, and training takes all the
CPU. A retrain that `label-export` queues starts as soon as that run ends. It serves rather than
polling a work pool for the same reason as the `monitor` service: there is one process and nothing
to provision.
"""

from datetime import timedelta

from prefect import serve

from cxr_flows.config import load_labeling_config
from cxr_flows.label_export import label_export_flow
from cxr_flows.labeling import TRAIN_DEPLOYMENT
from cxr_flows.train import train_flow


def main() -> None:
    cfg = load_labeling_config()
    # to_deployment is typed sync-or-async; outside a coroutine it is sync.
    serve(
        train_flow.to_deployment(name=TRAIN_DEPLOYMENT.split("/")[1]),  # type: ignore[arg-type]
        label_export_flow.to_deployment(  # type: ignore[arg-type]
            name="scheduled", interval=timedelta(seconds=cfg.export_interval_seconds)
        ),
        limit=1,
        webserver=True,
    )


if __name__ == "__main__":
    main()
