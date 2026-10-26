# 0001: RustFS as the S3 object store, behind one set of endpoint env vars

- Status: Accepted
- Date: 2026-10-24

## Context

Datasets (DVC remote), MLflow artifacts, model bundles, and raw inference uploads all need object storage that runs on-prem in Docker Compose. Every component speaks S3: boto3 in the backend and worker, MLflow's artifact store, DVC through s3fs, Label Studio's source storage, and aws-cli in the init job.

MinIO, the usual choice, stopped publishing community Docker images and moved features out of its open-source edition, which makes it a poor default for a repo people are meant to clone and run.

## Decision

Use RustFS (`rustfs/rustfs:1.0.3`) as the single S3 endpoint, with four buckets: `datasets`, `artifacts`, `inference`, and `mlflow`. `rustfs-init` creates them, plus a lifecycle rule that expires `inference/raw/`.

Every client reaches the store only through standard env vars, never a hard-coded URL:

| Variable | Set from `.env` |
| --- | --- |
| `AWS_ENDPOINT_URL` | `S3_ENDPOINT_URL` in containers, `S3_HOST_ENDPOINT_URL` on the host |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | `S3_ACCESS_KEY` / `S3_SECRET_KEY` |
| `AWS_DEFAULT_REGION` | `S3_REGION` |

boto3, aws-cli, and DVC all honor `AWS_ENDPOINT_URL` natively, so `.dvc/config` has no `endpointurl`. MLflow also gets `MLFLOW_S3_ENDPOINT_URL` set from the same value.

MLflow clients set `MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD=false` and `MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD=false`. With an S3 artifact store, the MLflow server otherwise hands out presigned URLs signed for its own in-network endpoint (`http://rustfs:9000`), which host-side clients can't resolve and hang on. Artifacts stream through MLflow instead; model bundles are small, so the cost is negligible.

## Consequences

- Swapping to MinIO, Ceph RGW, or AWS S3 means editing `.env` and nothing else. The integration tests in `tests/integration/` run unchanged against any of them.
- Containers and the host need different endpoint hostnames. That is why `.env` carries two endpoint values and the Makefile exports the host-side one.
- RustFS is younger than MinIO. Phase 0's integration tests (put/get, multipart, presigned URL, lifecycle, MLflow artifacts, DVC push/pull) are the guard, and all pass on 1.0.3.

## Alternatives considered

- **MinIO:** mature, but no maintained community image, and the licensing direction is unclear.
- **SeaweedFS / Garage:** workable S3 layers, but not tested against the features this project depends on (lifecycle rules, multipart, presigned URLs). Because of the env-var seam, either can be tried later by pointing `.env` at it and running the integration tests.
- **Local filesystem for everything:** simplest, but DVC remotes, MLflow artifacts, and Label Studio storage would each need a different setup, and the project would no longer show a realistic on-prem stack.
