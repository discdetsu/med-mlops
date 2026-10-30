"""Names shared by everything that touches the model registry: the flows that register
versions, the backend that moves aliases, and the workers that load bundles.

Stdlib only, so the backend and the worker can import it without numpy or torch.
"""

MODEL_NAME = "cxr-classifier"
CHAMPION_ALIAS = "champion"
CHALLENGER_ALIAS = "challenger"
ARTIFACTS_BUCKET = "artifacts"


def bundle_prefix(model_name: str, version: str) -> str:
    """Key prefix of a version's serving bundle in the artifacts bucket."""
    return f"models/{model_name}/{version}/"


def bundle_uri(model_name: str, version: str) -> str:
    return f"s3://{ARTIFACTS_BUCKET}/{bundle_prefix(model_name, version)}"


def split_s3_uri(uri: str) -> tuple[str, str]:
    """`s3://bucket/key/prefix/` → (`bucket`, `key/prefix/`)."""
    if not uri.startswith("s3://"):
        raise ValueError(f"not an s3:// URI: {uri!r}")
    bucket, _, key = uri.removeprefix("s3://").partition("/")
    if not bucket:
        raise ValueError(f"no bucket in {uri!r}")
    return bucket, key
