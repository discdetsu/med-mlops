# 0008: Monitoring compares each version's recent predictions with its own reference, and alerts through Prometheus without ever retraining

- Status: Accepted
- Date: 2026-11-01

## Context

Phase 4 has to show when the inputs or the outputs of the serving model move away from what it was validated on, how well it performs once reviewers label its predictions, and when the infrastructure fails, with alerts sent to Discord or Slack. Some constraints shape the design:

- Labels arrive late and rarely (reviewer feedback), so most of the signal has to come from unlabeled inputs and outputs.
- Embeddings from two model versions live in different spaces. Comparing version 3's embeddings with version 2's validation set measures the change of model, not drift.
- The drift reference already exists per version: `reference.parquet` in every bundle holds the validation set's embeddings, probabilities, and metadata as that version saw them (Phase 1).
- The plan names Evidently for drift and Prometheus, Alertmanager, and Grafana for alerting and dashboards. Drift is a batch computation, but everything else here is scraped.
- Monitoring must never start a retrain. Phase 5 retrains on new labels, not on drift.

## Decision

**What serving records.** Each prediction row stores the pooled features (GAP of the ONNX `features` output, `cxr_core.inference.pool`, the same vector `CamModel.pool` gave the reference), sent by the worker as 6-significant-digit floats in a JSON column, about 5 KB per ResNet18 prediction. Each job stores the image's sex, age, and view, from optional upload form fields, or from the DICOM header when the form leaves them out. A PNG without fields has none, and metadata drift then skips it.

**The monitor flow** (`cxr_flows.monitor`, Evidently in `cxr_flows.drift`, thresholds in `flows/src/cxr_flows/configs/monitor.yaml`) runs every 5 minutes as the `monitor` service under `full` (`flow.serve`), or once with `make monitor`. For every model version with recent predictions or feedback:

- **Window:** the version's newest 200 predictions from the last 60 minutes, each uploaded file counted once (by sha256). A window under 100 predictions isn't judged. A fixed number of predictions keeps statistical power the same at any traffic level, and the age limit lets an old shift age out. Deduplication matters for more than accounting: with the same file on both sides of Evidently's train/test split, the domain classifier memorizes it. On the first live run, 50 duplicates in an in-distribution window pushed the embedding score from 0.48 to 0.635, above the threshold.
- **Reference:** the version's own `reference.parquet`, 3,000 rows sampled with a fixed seed, so successive windows are scored against the same rows.
- **Scores, all "higher means more drift", each compared with its own threshold:**

  | Kind | Score | Threshold | Random windows of 100 | AP-only windows of 100 |
  | --- | --- | --- | --- | --- |
  | embedding | ROC AUC of Evidently's domain classifier (0.5 = indistinguishable) | 0.6 | 0.50 ± 0.045, max 0.57 | 0.70 ± 0.04 |
  | prediction | normed Wasserstein distance per class probability | 0.3 | largest class 0.17 | largest class 0.41 |
  | metadata | Jensen-Shannon distance (view, sex), normed Wasserstein (age) | 0.2, 0.2, 0.3 | view 0.02, sex 0.03, age 0.12 | view 0.54 |

  Calibrated on the v2 champion with the 4,000 cached train-split images, which come from other patients than the validation reference but the same distribution: 8 windows of each kind at sizes 50, 100, and 200. Below 100 images, random windows reach 0.56 (embedding) and 0.25 (prediction), hence the minimum window. Age is noisy at small sizes, hence its higher threshold. PCA before the classifier did not separate the two cases better.
- **Delayed AUROC:** each version's probabilities against the latest reviewer labels per job from the last 30 days, per class and macro, with a 95% percentile-bootstrap CI over jobs (`cxr_core.metrics.bootstrap_auroc`). Nothing is reported below 30 labeled predictions, or for a class without both a positive and a negative label. An "agree" answer is stored as the model's own thresholded predictions, so those rows separate perfectly by construction and all-agree feedback gives exactly 1.0. They still count, because a reviewer did endorse them and leaving them out would keep mostly disagreements, which biases the score the other way. `cxr_delayed_auroc_agree_share` reports their share per version, on the CXR · Model dashboard next to the AUROC.
- **Outputs:** the Evidently HTML plus a JSON summary go to `s3://artifacts/monitoring/<model>/<version>/<time>.*`, expiring after 30 days (`MONITORING_REPORT_RETENTION_DAYS`), and a row goes to `drift_reports`. An unchanged window (same job ids) reuses its last report instead of writing a new one. Every score, its threshold, the window size, the delayed AUROC and CI, the `@champion`, the interval, and `cxr_monitor_last_success_timestamp_seconds` go to the Pushgateway in one PUT. That replaces the whole group, so a version that leaves the window stops reporting scores, and a pass that fails pushes nothing.

**Table ownership: the same exception as ADR 0007.** Django owns `drift_reports` through the `monitoring` app's migration, and the flow only inserts, so the admin (Monitoring → Drift reports, with the Evidently report behind a link) works before the first pass. The report is a self-contained React page, served with a CSP that allows inline scripts only inside `sandbox allow-scripts`. The page then runs in an opaque origin, away from the admin's cookies and DOM; headless Chromium showed it needs neither `eval` nor the network.

**System metrics.** Prometheus scrapes the backend's `/metrics`, every worker's `:9100/metrics` (found through Docker DNS, so `--scale worker=N` needs no config: jobs by outcome including retries, job duration, the version each worker serves, model swaps), RabbitMQ's node totals, and its per-queue `/metrics/detailed` for the DLQ and the inference queue's consumers.

**Alerts** (`compose/prometheus/rules/cxr.yml`, unit-tested with promtool: `make test-rules`):

- `CxrApiErrorRate`: more than 5% 5xx over 5 minutes, with at least one request every 10 s.
- `CxrBackendDown`: the backend can't be scraped.
- `CxrWorkerMissing`: no worker up. Replicas leave Docker's DNS when they stop, so this has to be `absent(up == 1)`, not `up == 0`.
- `CxrDeadLetterQueueNotEmpty`.
- `CxrDriftDetected`: `cxr_drift_score > cxr_drift_threshold`. The threshold comes from the same push, so monitor.yaml is the only place it is set.
- `CxrMonitorStale`: the last success is more than 3 pushed intervals old.
- `CxrMonitorNeverRan`.

Alertmanager groups alerts by alertname and model version. Its entrypoint picks the Discord or Slack receiver from `DISCORD_WEBHOOK_URL` / `SLACK_WEBHOOK_URL` in `.env` and hands the URL over as a file, because Alertmanager reads no env vars in its config. With neither set, alerts only show in its UI.

**Dashboards as code:** `compose/grafana/dashboards/` holds CXR · Serving, CXR · Queue, and CXR · Model, the last with a `model_version` variable, drift scores against their thresholds, and delayed AUROC with its CI.

**The monitor image** (`flows/Dockerfile`) is the flows package without torch, torchvision, and timm (`uv sync --no-install-package`), and the build fails if importing the monitor flow ever pulls in torch.

**Evidently on pandas 3.** Evidently 0.7.23, the latest release, imports `pandas.core.dtypes.common.infer_dtype_from_object`, which pandas 3 removed. Only a legacy test class uses it, and none of the metrics here do. `cxr_flows.drift` restores the pandas 2 behaviour before importing Evidently, and its unit tests run every metric used.

## Consequences

- Drift alerts mean "this version's recent inputs or outputs don't look like its validation set". They don't mean "the model got worse"; only the delayed AUROC says that, and only once reviewers have labeled 30 of its predictions.
- Delayed AUROC is optimistic in proportion to the agree share, and near a share of 1 it says nothing about the model. Trust it most when the share is low. Reviewers who see the prediction before answering also anchor on it, which no change to this calculation removes.
- After a promotion, the new version is judged against its own reference from its first 100 predictions. The old version's scores disappear once its predictions age out of the 60-minute window.
- Uploads without metadata (plain PNGs from the UI) still get embedding and prediction drift, but no metadata drift. `scripts/replay_traffic.py` sends NIH metadata as form fields, as a PACS integration would.
- `predictions` grows by about 5 KB per row for the embedding. A demo stack doesn't notice; a busy one would want a retention job or embeddings in object storage.
- A pass costs about 10 s of CPU per judged version (mostly Evidently's domain classifier and HTML), plus a 44 MB reference download per version the first time. The `monitor` container peaks at about 800 MB during a pass.
- The stale alert follows `interval_seconds` in monitor.yaml through the pushed `cxr_monitor_interval_seconds`. At the default 300 s it fires about 16 minutes after the last success.
- The pandas shim has to go when Evidently supports pandas 3; until then, upgrading Evidently needs the drift tests to pass.

## Alternatives considered

- **One reference for all versions (the current `@champion`'s):** embeddings from different versions aren't comparable, and right after a promotion every window would "drift".
- **Time windows (everything from the last N minutes):** the alert would depend on traffic. A quiet hour gives a window too small to judge, a busy one a window that dilutes a short shift.
- **Evidently's own UI and workspace:** one more service and store to run, and alerting would still have to go through Prometheus.
- **Drift computed in Django or the worker per request:** puts Evidently and scikit-learn in serving, and drift is a property of a window, not of one request.
- **Grafana alerting instead of Alertmanager** (cut list item 3): workable, but the rules would live in Grafana's provisioning format, which promtool can't unit-test.
- **Embeddings as `.npy` objects in S3 instead of a JSON column:** the monitor would issue one GET per prediction per pass, instead of one SQL query.
- **Pinning pandas below 3 for Evidently:** a downgrade under the Phase 1 data code, which runs on pandas 3's string dtype and copy-on-write.
