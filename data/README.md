# data/

DVC-tracked; the remote is `s3://datasets/dvc` on RustFS. Only `.dvc` pointer files are committed. No images go into git.

| Path | Contents | Written by |
| --- | --- | --- |
| `nih/` | NIH ChestX-ray14 subset with canonical metadata, patient-level train/val split | `make seed` (Phase 1 `ingest` flow) |
| `test_locked/` | Frozen locked test set, used by evaluation and the promotion gate | `ingest`, once |
| `curated/` | Reviewer labels exported from Label Studio | Phase 5 export job; the only place new labels may go |
| `golden/` | A few PNG and DICOM files (including `MONOCHROME1`) for parity and smoke tests | Phase 1 |

No patient may appear in more than one split or in both `test_locked/` and any other path. The Pandera schema enforces this.
