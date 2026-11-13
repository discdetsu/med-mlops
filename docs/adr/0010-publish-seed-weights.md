# 0010: Publish the seed model as a GitHub Release, with the NIH attribution

- Status: Accepted
- Date: 2026-11-12

## Context

The `core` demo has to work from a fresh clone without training anything, and training needs the NIH images plus the frozen `test_locked` set, which a fresh clone doesn't have (ADR 0003). So the first `@champion` has to come from somewhere outside the repo. `make seed-model` already downloads a seed bundle (ONNX model, head weights, thresholds, drift reference) and checks it against its `.sha256`, but the `seed-v2` release it points at was held back until the dataset's terms were checked. Without it the CI smoke test skips, and M2 isn't met.

The terms, as published with the dataset (quoted from the Google Cloud copy of the NIH dataset page, since nih.gov refused automated fetches): "There are no restrictions on the use of the NIH chest x-ray images." Three attribution requirements come with that: link to the NIH download site (https://nihcc.app.box.com/v/ChestXray-NIHCC), cite the CVPR 2017 paper (Wang et al.), and acknowledge the NIH Clinical Center as the data provider. The terms say nothing specific about trained weights.

## Decision

Publish `cxr-seed-bundle-v2.tar.gz` and its `.sha256` as the GitHub Release `seed-v2`. The release notes, the README, and the model card carry all three attribution items and the "not a medical device" disclaimer.

- The bundle has no images. Its drift reference (`reference.parquet`) holds, for the 16,166 validation images: NIH image IDs, sex, age, view, the model's probabilities, and 512-value embeddings. All of it is derived from public, de-identified NIH data.
- Releases are immutable by convention: a new seed model gets a new tag (`seed-v3`, …) and `SEED_BUNDLE` in the Makefile moves to it, so old clones keep working.
- The code stays MIT. The weights are distributed under the same terms as the data they came from, with the attribution above, and the README says so separately from the code license.

## Consequences

- A fresh clone runs `make up && make seed-model` with nothing else, and the CI smoke test runs on every push to `main`.
- The bundle is about 85 MB, half of it the drift reference. That's fine for a release asset, but too big to commit.
- If NIH's terms change, or a takedown request comes in, the release is deleted and the README falls back to `make bootstrap-champion`. Nothing else depends on the release existing.
- "Unrestricted use" doesn't make the model fit for clinical use. The disclaimer is the guard here, not the license.

## Alternatives considered

- **Keep the weights local and make users train:** needs the full NIH download (45 GB), a `make seed`, and about 30 minutes of CPU training, so the demo stops being a three-command demo, which defeats the point of `core`.
- **Host the bundle on Hugging Face Hub:** a model card and discovery for free, but a second account and service for a single file. GitHub Releases sits next to the code and `gh` already handles it.
- **Ship a model trained on synthetic or other data:** avoids the question, but the heatmaps and metrics would mean nothing.
