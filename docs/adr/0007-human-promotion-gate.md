# 0007: Promotion is a paired-bootstrap gate on the locked test set, then a person approves in Django admin

- Status: Accepted
- Date: 2026-10-31

## Context

A new version must not replace `@champion` because it scored higher on one test run. The locked test set (ADR 0003) has 16,615 images from 4,670 patients, so two models' AUROCs differ by a few thousandths from noise alone, and a model can improve on average while getting worse for AP views or older patients, or while its probabilities drift out of calibration. The plan also requires a person to sign off on every promotion, with a record of who, when, and why, and a rollback that is as easy as the approval. Every worker has to end up on the same version without dropping jobs (ADR 0005 already covers the swap).

## Decision

**The gate** is the `gate` Prefect flow (`cxr_flows.gate`; the decision itself is `cxr_flows.gating`). The `train` flow runs it after registering a version, and `make gate VERSION=N` re-runs it.

- Both versions score the same locked test set the way serving does: `cxr_core.preprocess` from the raw files, then each version's own ONNX bundle through `cxr_core.inference`, at its own image size. The gate compares the artifacts that would serve, not the PyTorch models.
- **Paired bootstrap over patients** (`cxr_core.metrics.paired_bootstrap_auroc`). Each resample draws patients with replacement and applies the same draw to both models, so the CI of challenger − champion reflects only how the models differ. Whole patients are resampled because one patient's follow-up images are not independent. Each model's scores are sorted once, and every resample is then a cumulative sum over case weights, so 1,000 resamples over 8 groups take about 4 s.
- **It passes when all of these hold** (thresholds in `flows/src/cxr_flows/configs/gate.yaml`):
  - the lower bound of the 95% CI of macro ΔAUROC is at least −`margin` (0.02), a non-inferiority test;
  - no subgroup (sex, view, age band) with at least `min_subgroup_size` (100) images has a macro ΔAUROC point estimate below −`margin`, or no ΔAUROC at all (no class with both positives and negatives there), since the gate then can't show the challenger is fine on it. Subgroup CIs are reported but not judged: they are too wide for a per-subgroup non-inferiority test to pass any retrain;
  - pooled ECE rises by at most `max_ece_increase` (0.01);
  - the challenger passed E2E parity.
- The report (JSON + HTML) goes next to the version's bundle (`s3://artifacts/models/<name>/<v>/gate/vs-champion-v<c>.*`) and into its MLflow run under `gate/`. The verdict is also set as version tags (`gate_passed`, `gate_champion_version`, ...), and the flow upserts a row into `model_candidates`.

**The decision** is Django's (`cxr_backend.registry.promotion`). The admin's buttons and the `promote` / `rollback` management commands call the same functions.

- *Approve* needs the `registry.promote_model` permission, a reason, a candidate that passed and is still pending, and a `@champion` equal to the version the gate compared it with. Otherwise the comparison is stale and the candidate has to be re-gated. Approval records a `model_approvals` row, moves `@champion` to vN and `@previous` to the old champion, then drops `@challenger` if it pointed at vN and publishes `champion_changed` on `model.events`.
- *Reject* closes a pending or gate-failed candidate and drops `@challenger` if it pointed at it.
- *Rollback* moves `@champion` back to `@previous` and clears `@previous`, so rollback is one step. A second rollback is refused until another approval, instead of flipping between two versions. It is also refused when `@previous` already equals `@champion`.
- MLflow aliases stay the source of truth for what serves. Approvals and rollbacks are serialized with a Postgres advisory lock. Each action writes its database rows first and moves `@champion` and `@previous` last, in the same transaction. If either alias move fails, the other is put back and the rows roll back. If the commit fails after the aliases moved, both are put back. So the serving version and the records stay in step. Dropping `@challenger`, the `promotion_status` tag, and the `model.events` publish come after the commit and are best-effort. If the publish fails, the approval records `event_published = false`, the admin warns, and workers catch up when they reconnect (ADR 0005).
- **`auto_promote`** (off by default, for the scripted demo in Phase 5): the gate marks the candidate, and `make train` ends with `promote --auto`, which approves it through the same function, with actor `auto_promote`. It only considers `@challenger`, the version that run just registered (or an explicit `promote --auto VERSION`), so an older marked candidate is never promoted in its place. A train run started from a Prefect deployment does not run `promote --auto`; Phase 5 adds that step to its retrain flow.

**Table ownership: the exception to ADR 0006.** Django owns `model_candidates` and `model_approvals` through its migrations. The flows still own their dataset tables, but the gate flow inserts into `model_candidates`, whose schema it does not own. The admin has to list candidates and change their status before any gate has run, so the table must exist from the backend's first start. A flow-owned table would not exist until the first gate run.

## Consequences

- A promotion is never more than a click, and never less than a reason, a user, and a current comparison on record.
- Re-gating a version replaces its gate result and reopens the decision. Earlier decisions stay in `model_approvals`.
- The gate scores the 16.6k-image test set twice, about 3 minutes on a laptop CPU at 128 px (the tensor cache is shared with training), and adds that to every `make train`.
- The gate flow's insert has to keep up with Django's migrations for `model_candidates`. `tests/integration/test_promotion_db.py` checks it against the real schema.
- The non-inferiority margin and the subgroup rule are judgement calls. The margin is 0.02, chosen from the first real comparison (v1 vs v2): the 95% CI of macro ΔAUROC was ±0.009 around the point estimate, and the subgroup CIs were ±0.008 (PA) to ±0.019 (age 60+), each subgroup with at least 3,965 images. So an equally good retrain has a macro lower bound near −0.009 and passes with room to spare, while one that is truly 0.02 worse lands near −0.029 and fails. A subgroup point estimate falls below −0.02 by noise alone only about two standard deviations out. A margin of 0.01 would put an equally good retrain on the line and fail it on unlucky resamples; 0.03 or more would let real regressions through. A version up to about 0.01 worse can still pass, which is acceptable because a person approves every promotion with the report in front of them. Retrains that differ from the champion only by seed or data refresh should give narrower CIs than this pair, which makes 0.02 more lenient, not stricter. Re-check the widths if the locked test set changes size. The first real run caught a version whose AUROC was better but whose ECE rose from 0.006 to 0.086: it had been trained on a class-balanced subset, so its probabilities were miscalibrated at NIH prevalence.
- `make train`, `make promote`, and `make rollback` start a one-off backend container (`docker compose run`), so the core stack has to be up.

## Alternatives considered

- **Compare the AUROCs from each version's own training run:** older versions may have been evaluated on a different locked test set (it was rebuilt once), and unpaired CIs are much wider than paired ones.
- **Bootstrap over images instead of patients:** narrower CIs that overstate certainty, because a patient's follow-up images move together.
- **Judge every subgroup with its own CI lower bound:** subgroup CIs are 2 to 3 times wider than the macro one, so almost every retrain would fail.
- **Promote in the flow (MLflow alias + event), with Django only showing it:** two places would change `@champion`, and the record of who approved would live outside the app that does authentication.
- **A flow-owned `model_candidates` read by an unmanaged Django model (the ADR 0006 pattern):** the admin page would error until the first gate run created the table, and Django would still have to write status into a table it doesn't own.
- **Rollback as a toggle between `@champion` and `@previous`:** a second click would silently re-promote the version just rolled back.
