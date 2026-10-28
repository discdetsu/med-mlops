# 0003: Hash-based patient split, a frozen locked test set, and content-addressed dataset revs

- Status: Accepted
- Date: 2026-10-27

## Context

The promotion gate (Phase 3) compares `@champion` and `@challenger` on the same locked test set, and the labeling loop (Phase 5) keeps adding data. Both only make sense if:

- no patient is in more than one of train, val, and test, since NIH has patients with dozens of follow-up images;
- the locked test set never changes, whatever gets re-ingested or labeled later;
- every model version can name the exact data it was trained and evaluated on.

## Decision

**Split by a hash of the patient ID.** `sha256("cxr-split-v1:<patient_id>") mod 100` gives `train` below 70, `val` below 85, and `test_locked` otherwise. A patient's split depends only on their ID, not on which other patients were sampled, so it stays the same across subset sizes, re-ingests, and reviewer labels arriving in Phase 5. Subset sampling (`select_subset`) always draws from the full NIH table with a fixed seed, so the same arguments always give the same images.

**One DVC path per role, each with its own rule.**

| Path | Contents | Rule |
| --- | --- | --- |
| `data/nih/` | train + val images, `metadata.csv` | rewritten by `ingest` |
| `data/test_locked/` | locked test images, `metadata.csv` | created by the first `ingest`, then frozen |
| `data/golden/` | 3 test_locked images as PNG + DICOM MONOCHROME2 + MONOCHROME1 | created with test_locked, then frozen |
| `data/curated/` | reviewer labels (Phase 5) | the only place new labels may go |

Once `data/test_locked.dvc` exists, `ingest` pulls and checks test_locked and golden but never rewrites them, and drops sampled images from test_locked patients. The `train` flow refuses to run if nih, test_locked, or golden differs from its `.dvc` pointer. Pandera checks the canonical schema, image files and sizes, and one split per patient; `check_no_shared_patients` checks that no patient is in both `data/nih` and `data/test_locked`.

**`dataset_rev` is content-addressed:** the first 12 hex digits of sha256 over the sorted `path:md5` lines of `data/nih` and `data/test_locked`. The same data always gets the same rev, whatever git commit or machine produced it. Each model version is tagged with the rev, the per-path md5s, and the git commit, and `--dataset-rev` makes `train` fail if the workspace holds different data.

**Metadata lives in the app database too.** `ingest` records each version in `dataset_versions` and its membership in `dataset_images` (`dataset_rev`, `dvc_path`, `image_id`, `file`), so any version's images can be listed later. `images` and `labels` (long format, with `label_source`) hold one row per image and are upserted, because an image's metadata and NIH labels don't change between versions. The flows own these tables and create them idempotently; Django reads them through unmanaged models from Phase 2. Their names can't clash with Django's tables, which carry an app-label prefix.

## Consequences

- Growing the subset (`make seed SEED_ARGS="--positives-per-class 2000 …"`) changes `data/nih` and the rev, but not the locked test set, so old and new models stay comparable at the gate.
- The test split's size is fixed at the first ingest. To change it on purpose, delete `data/test_locked*` and `data/golden*`; all earlier gate results are then incomparable.
- The bootstrap CIs in evaluation resample images, not patients, so they are slightly too narrow when a patient has several test images. Accepted for the demo; the gate in Phase 3 can resample patients.

## Alternatives considered

- **NIH's official `train_val_list.txt` / `test_list.txt`:** patient-disjoint, but defined over the full 112k images. The default `make seed` takes a 4k subset, and new reviewer labels need a split rule anyway.
- **Seeded shuffle of the sampled patients:** a patient's split would change with the subset size, so test patients could leak into training after a re-ingest.
- **Git commit of the `.dvc` files as the rev:** not available until someone commits, and two commits can hold identical data.
