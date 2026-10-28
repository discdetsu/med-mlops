# data/

DVC-tracked; the remote is `s3://datasets/dvc` on RustFS. Only `.dvc` pointer files are committed. No images go into git.

| Path | Contents | Written by |
| --- | --- | --- |
| `nih/` | NIH ChestX-ray14 subset: `images/*.png` + `metadata.csv`, train and val splits | `make seed` (Prefect `ingest` flow), on every run |
| `test_locked/` | Frozen locked test set: `images/*.png` + `metadata.csv`, used by evaluation and the promotion gate | first `make seed` only; frozen afterwards |
| `golden/` | 3 test_locked images, each as PNG, DICOM `MONOCHROME2`, and DICOM `MONOCHROME1`, plus `manifest.csv`; used by the E2E parity test | together with `test_locked/`; frozen afterwards |
| `curated/` | Reviewer labels exported from Label Studio | Phase 5 export job; the only place new labels may go |

By default `make seed` downloads only the sampled files from Kaggle (`nih-chest-xrays/data`, public, no token needed), one at a time, into `.cache/cxr/kaggle`, so a re-run after `dvc checkout` downloads nothing. Grow the subset with `make seed SEED_ARGS="--positives-per-class 2000 --negatives 6000"`, or take every image with `--positives-per-class all --negatives all`. That changes `nih/` but never `test_locked/`.

For large subsets, use the full archive instead of per-file downloads:

| Command | Source |
| --- | --- |
| `make seed SEED_ARGS="--from-archive"` | downloads Kaggle's whole archive once (45 GB; about 90 GB free needed while extracting; resumable) into `.cache/cxr/kagglehub` |
| `make seed SEED_ARGS="--from-archive /path/to/dataset"` | an already extracted copy: `Data_Entry_2017.csv` plus `images_001/` … `images_012/`, as on Kaggle or NIH's own release |

Files from a folder you pass are copied, never hard-linked, because DVC makes its cache files read-only and a hard link would change your copy too. The same arguments select the same images from any source, so the resulting `dataset_rev` doesn't depend on where the files came from.

## Canonical schema

One row per image, the same columns in every `metadata.csv`. In Postgres they are split across `images`, `labels`, and `dataset_images` (which version holds which image, and its `file`):

`image_id`, `file` (relative to the dataset directory), `patient_id`, `patient_sex` (M/F), `patient_age` (empty if NIH's value is impossible), `view_position` (PA/AP), `follow_up`, `width`, `height` (measured from the file), `split` (train/val/test_locked), `label_source` (nih_nlp/reviewer), then one 0/1 column per class: `Atelectasis`, `Cardiomegaly`, `Effusion`, `Mass`, `Pneumothorax`.

## Rules

- The split is a hash of `patient_id` (70/15/15), so a patient is always in exactly one split, whatever subset is drawn.
- No patient may appear in both `nih/` and `test_locked/`, or later in `curated/` and `test_locked/`.
- Pandera checks the schema, the files and their sizes, and the first rule; `check_no_shared_patients` checks the second. Both run in `ingest`, again in `train`, and in CI on synthetic data.

See [ADR 0003](../docs/adr/0003-dataset-versions-and-locked-test.md) for the reasoning and for what `dataset_rev` means.
