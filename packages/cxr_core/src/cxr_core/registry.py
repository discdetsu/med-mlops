"""Names shared by everything that touches the model registry: the flows that register
versions, the backend that moves aliases, and the workers that load bundles.

Stdlib only, so the backend and the worker can import it without numpy or torch.
"""

MODEL_NAME = "cxr-classifier"
CHAMPION_ALIAS = "champion"
CHALLENGER_ALIAS = "challenger"
# The champion before the last promotion; rollback moves @champion back to it.
PREVIOUS_ALIAS = "previous"
ARTIFACTS_BUCKET = "artifacts"

# Promotion candidates (ADR 0007): Django owns the table and its migrations; the gate flow
# inserts one row per gated version with one of the first two statuses, and the admin's
# approve / reject / rollback actions move it to the others.
CANDIDATES_TABLE = "model_candidates"
STATUS_PENDING = "pending"
STATUS_GATE_FAILED = "gate_failed"
STATUS_APPROVED = "approved"
STATUS_REJECTED = "rejected"
STATUS_ROLLED_BACK = "rolled_back"


def bundle_prefix(model_name: str, version: str) -> str:
    """Key prefix of a version's serving bundle in the artifacts bucket."""
    return f"models/{model_name}/{version}/"


def bundle_uri(model_name: str, version: str) -> str:
    return f"s3://{ARTIFACTS_BUCKET}/{bundle_prefix(model_name, version)}"


def gate_report_prefix(model_name: str, version: str) -> str:
    """Key prefix of a version's gate reports, next to its bundle (workers never read it)."""
    return f"{bundle_prefix(model_name, version)}gate/"


def split_s3_uri(uri: str) -> tuple[str, str]:
    """`s3://bucket/key/prefix/` → (`bucket`, `key/prefix/`)."""
    if not uri.startswith("s3://"):
        raise ValueError(f"not an s3:// URI: {uri!r}")
    bucket, _, key = uri.removeprefix("s3://").partition("/")
    if not bucket:
        raise ValueError(f"no bucket in {uri!r}")
    return bucket, key
