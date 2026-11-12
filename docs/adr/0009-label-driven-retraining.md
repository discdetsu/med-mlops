# 0009: Reviewer labels, not drift, start a retrain, and they reach training only through a checked, versioned path

- Status: Accepted
- Date: 2026-11-05

## Context

Phase 5 closes the loop: drifted cases and disagreements reach Label Studio, labels come back as a new dataset version, and enough new labels start a retrain. Some constraints shape the design:

- Drift says the inputs moved, not that a new model would be better, and a retrain on the same data can't fix it anyway. Retraining needs new labels.
- Reviewer labels must never reach the locked test set, or every gate comparison after that is compromised (ADR 0003). An upload had no patient ID until now, so nothing could check this.
- Reviewer time is the scarce resource. A label that could never be used shouldn't be asked for.
- Selected cases are biased towards what the model finds hard, and evaluating on them says little without a slice that stays out of training.
- `data/` and its DVC cache are tens of GB, and `make train` already works on the workspace on the host.

## Decision

**Selection** (`cxr_flows.labeling`, run by the monitor flow, thresholds in `flows/src/cxr_flows/configs/labeling.yaml`). Each pass sends, up to `max_tasks_per_pass` (100):

1. every prediction whose latest reviewer feedback differs from its thresholded prediction;
2. from each window that drifted, the 25 farthest from the version's reference: distance from the centroid, per-dimension in reference standard deviations, so a feature with a wide range doesn't dominate;
3. a random sample of 5 from each new window, to limit selection bias, seeded by the window so a re-run picks the same ones.

The cap trims disagreements first, newest kept. They are selected again on the next pass, whereas the other two come from a window once. On the first live run with most jobs given feedback, 96 disagreements filled the cap and left no random sample. If drift and the random sample alone exceed the cap, which takes four drifted windows at these settings, drift loses each window's nearest cases evenly, and the random sample is kept to the last.

One task per file (by sha256), like the drift windows. Cases without a patient ID, or whose patient hashes to `test_locked`, aren't sent: their labels could never be used. Tasks show the image and the choices, with no model output and no selection reason, because a reviewer who sees the prediction anchors on it (ADR 0008). A window that hasn't changed since the last pass adds nothing, since that pass already picked from it. If a pass can't reach Label Studio, disagreements are picked up by the next pass and drift cases by the next new window.

**No duplicates.** A Postgres advisory lock serializes the monitor and export flows, whichever process runs them, and a case is recorded in `labeling_tasks` only after Label Studio has it. If a pass dies in between, the next one finds the task through its `sha256` in Label Studio's task list and records it instead of creating another. A Label Studio outage is logged and doesn't fail the monitor pass, because `CxrMonitorStale` must keep meaning that monitoring stopped.

**Label Studio** holds a multi-label project over the canonical classes, with a required choice and an explicit "None of these", so an empty answer can only mean nobody answered. Its storage points at the `inference` bucket, not the DVC remote, where file names are hashes. Tasks use the display PNG the worker wrote, because browsers can't draw DICOM. Label Studio proxies the images (`presign: false`), since a presigned URL would name `http://rustfs:9000`, which a browser can't reach. Two of its 1.23 behaviours decided the rest: its SSRF guard refuses a private S3 endpoint set through the API but trusts the operator's `S3_ENDPOINT` env var, and legacy API tokens stay off unless the organization was created with a CLI flag. So the flows log in with the admin account already in `.env`, which works on any existing database and adds no secret.

**Export** (`cxr_flows.label_export`, the `label-export` flow, every 15 minutes in `flows-worker` or `make label-export`):

- The latest finished annotation per task becomes {class: 0/1}. Skipped, empty, or contradictory answers leave the task open.
- The raw upload, not the display PNG, goes through Pandera with the canonical schema, the file's real size, and a `test_locked` patient check. **A row that fails is quarantined, not fatal:** it stays on its `labeling_tasks` row with the answer and the reason, and never reaches `labels`, `images`, or `data/curated/`. An expired upload or missing sex or view is quarantined the same way. Any other S3 error, such as an outage or a permission problem, fails the pass and leaves the tasks open for the next one. Metadata is never guessed. `train` re-checks `check_no_shared_patients` on every run, so a hand-edited `metadata.csv` fails there.
- The rest is written to `data/curated/` under a content-based image ID, then `dvc add`, `dvc push`, and `load_dataset_version`, the same load an ingest uses, which writes `labels` with `label_source=reviewer`. A rerun after a crash at any point yields the same rev, because `metadata.csv` is sorted and the IDs come from content.
- A changed answer updates its row and makes a new rev, but doesn't count as a new label.

**The patient ID** is a new optional upload field, filled from the DICOM header when the form leaves it out, and kept out of the public API's responses. An ID longer than 64 characters is dropped, not cut, because a cut ID could name another patient.

**Where a label goes** follows the patient, as in ADR 0003 (`curated_split`):

| Patient's hash bucket | Curated split | Why |
| --- | --- | --- |
| `test_locked` | refused (quarantine) | The bucket stays reserved for the locked test set, whether or not this patient is in its frozen subset |
| `val` | `recent` | Curated data never changes the validation set behind thresholds, early stopping, and the drift reference |
| `train`, 20% by a second salted hash | `recent` | The held-back slice: evaluated, never trained on |
| `train`, the other 80% | `train` | Trains |

**Training** adds curated `train` images after the config's NIH sample, so that sample stays the same across revs, and removes NIH images of `recent` patients from training, so "never for training" holds for the patient, not just the file. A merged memmap in the tensor cache avoids re-preprocessing thousands of images for every rev. `recent` gets a section in the evaluation report and a `recent_macro_auroc` metric. `data/curated` joins `dataset_rev` from its first export; before that, the rev is the same as it was, so existing models keep theirs.

**`should_retrain()`** (`cxr_flows.labeling`) is a pure function of revs and counts, and drift isn't one of its inputs. It says yes on a new rev with at least `min_new_labels` curated **training** images that the newest registered version didn't have. Counting from the newest version rather than from the last trigger means a retrain that fails doesn't lose its labels; a rev is only ever started once, so it doesn't loop either. Prefect enforces that with an idempotency key on the rev, so a crash between starting the run and recording it gets the same run back on the next pass. Each changed outcome is a row in `retrain_decisions`, and every pass logs its decision.

**While a retrain is unfinished, exports wait.** The worker's `limit=1` only covers runs it serves, and `make label-export` or `make demo` run on the host. So an export checks the state of the last run it started, under the labeling lock, and writes and decides nothing until that run ends. Otherwise it could change `data/curated` under the run, or make a new rev and queue a second retrain for one more label. A run unfinished after `retrain_timeout_hours` (12) is presumed dead, because a killed worker leaves it `Running` for good.

**N is 50.** It is a throttle on cost, not a statistical threshold: a retrain takes about 30 minutes of CPU and a person's approval, so it should batch a reviewer's session rather than follow every few labels. Fifty is about 1% of the demo profile's training set, roughly what one demo round produces. It says nothing about whether the model will improve; the gate decides that. Raise it if retrains become routine.

**The worker** (`flows-worker`, `cxr_flows.deployments`) serves `train/retrain` and `label-export/scheduled` with `limit=1`: both write the DVC workspace, and training takes all the CPU. It uses `serve` like the `monitor` service, because there is one process and no work pool to provision. It bind-mounts the repo and runs as the host user (`HOST_UID`), so it sees the same `data/`, `.dvc/cache`, and pointers as `make train`, and what it writes stays the host's. `should_retrain()` passes `dataset_rev`, so a run that starts after the data has moved on fails instead of training on something else.

**Promotion from a deployment.** No make target follows a deployment run, so the flow calls a new `POST /internal/auto-promote`, which runs the same `promotion.auto_promote` as `promote --auto` and is a no-op unless the gate marked the version. `auto_promote` can be set for one run, so `make demo` uses it without touching the default in `gate.yaml`, and an ordinary retrain still waits for a person.

**Table ownership: the same exception as ADR 0007 and 0008.** Django owns `labeling_tasks` and `retrain_decisions` through the `labeling` app, and the admin lists them read-only. The flows keep owning the dataset tables; Django still doesn't read those, because the task and decision rows already answer what the admin needs (superseding ADR 0006's guess that this phase would bring the first unmanaged models).

**`make demo`** (`scripts/demo.py`) plays the whole loop on the `full` stack: AP views from the train split that the champion didn't train on, then a real monitor pass, the real NIH labels posted as annotations through Label Studio's API, a real export, and the real train deployment, and it checks that a new upload is predicted by the new version on the new rev.

## Consequences

- A prediction now traces to a rev that includes reviewer labels, and `labeling_tasks` traces each label back to its job. `dataset_versions.dvc_md5` can rebuild `data/curated.dvc` for any rev.
- Uploads without a patient ID, which includes the UI's plain PNGs, can't enter the loop, even with feedback. That is deliberate: with no way to check the locked test set, the safe answer is no. A DICOM carries its own ID, and an integration can send the form field.
- A patient's labels are thrown away if their hash bucket is `test_locked`, about 15% of real traffic. That is the price of keeping the bucket's meaning stable.
- `recent` is drawn from selected cases, so it is harder than the population. Compare it between versions; don't read it as an estimate of real-world performance.
- With NIH traffic, a `val`-bucket patient in `recent` also has images in the validation set behind the model's early stopping. Real traffic has no such overlap, and `make demo` sends only `train`-bucket patients.
- The worker writes `data/curated.dvc` in the working tree and DVC stages it (`core.autostage`), but nothing commits it, so a run's `git_commit` tag can end in `-dirty`. The `dvc_md5.*` tags still say exactly which data it used.
- A scheduled export needs the `full` stack running on a workspace that holds `data/`. Exports and retrains run one at a time, so a label export can wait up to a retrain's length.
- Each new rev holding data/curated adds a merged tensor file of about 800 MB to the disposable cache at the demo profile's size.
- The flows log in with Label Studio's admin account, so `LABEL_STUDIO_USERNAME` and `LABEL_STUDIO_PASSWORD` must stay in step with its database; changing them in `.env` alone breaks the login, with an error that says so.

## Alternatives considered

- **Retrain on drift:** the plan rules it out, and a drift alert with no new labels would retrain on identical data and produce the same model.
- **Fail the whole export on one bad row:** a single expired upload would block every other label until someone intervened.
- **Quarantine as a DVC path:** versioning data that must never be used. The row holds the answer and the reason, and the raw file stays in `inference/raw/` until the lifecycle rule removes it.
- **Pre-annotate tasks with the model's predictions:** faster labeling, but the labels would lean towards the model, which is what this loop has to correct.
- **A copy of the workspace in a named volume, or `dvc pull` inside the container:** another 43 GB on disk, and pointers that drift apart from the host's.
- **A Prefect work pool with `prefect worker start`:** more moving parts (pool, deployment registration, entrypoints) for the same single process.
- **`python -m cxr_backend promote --auto` from the worker:** would put Django in the training image.
- **Personal access tokens or a compose flag for legacy tokens in Label Studio:** tokens need a login to create anyway, and the flag only applies to a new organization, so an existing `labelstudio-data` volume would break.
- **The Label Studio SDK:** a large dependency for six API calls, pinned to its own release cycle on a Python 3.14 stack.
- **Comparing curated images with NIH by content at training time:** catches a re-upload of an image already in the sample, which real traffic doesn't do, at the cost of hashing tens of thousands of files. `make demo` avoids those images instead.
