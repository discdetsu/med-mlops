# Architecture decision records

One file per decision, written when the decision is made. Copy `template.md`, take the next number, and never rewrite an accepted ADR. Supersede it with a new one instead.

| ADR | Decision | Status |
| --- | --- | --- |
| [0001](0001-rustfs-object-storage.md) | RustFS as the S3 object store behind one set of endpoint env vars | Accepted |
| [0002](0002-compose-core-full-profiles.md) | Two Compose profiles: `core` and `full` | Accepted |
| [0003](0003-dataset-versions-and-locked-test.md) | Hash-based patient split, frozen locked test set, content-addressed dataset revs | Accepted; partly superseded by 0006 |
| [0004](0004-onnx-gap-linear-cam.md) | ONNX with `logits` + `features`; GAP + linear heads only, so CAM is exact | Accepted |
| [0005](0005-async-inference-queue.md) | Async inference via RabbitMQ; solo-pool workers swap models between jobs; poison jobs go to the DLQ | Accepted |
| [0006](0006-flow-tables-in-django-on-demand.md) | Django reads the flows' tables only once a feature needs them; flows and backend share one table namespace | Accepted; Django owns `model_candidates` per 0007 and `labeling_tasks` per 0009 |
| [0007](0007-human-promotion-gate.md) | Paired patient-level bootstrap gate on the locked test set, then approval in Django admin; rollback to `@previous` | Accepted |
| [0008](0008-monitoring-drift-and-alerts.md) | Each version's recent predictions vs its own drift reference with Evidently, delayed AUROC from feedback, alerts through Prometheus and Alertmanager; never retrains | Accepted |
| [0009](0009-label-driven-retraining.md) | New labels start retraining, not drift; labels reach training only through Pandera checks, quarantine, and a new DVC rev; `flows-worker` runs the deployment | Accepted |
| [0010](0010-publish-seed-weights.md) | Publish the seed model as the `seed-v2` GitHub Release under the NIH data terms, with their attribution | Accepted |
