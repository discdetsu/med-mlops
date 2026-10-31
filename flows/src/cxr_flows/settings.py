"""Paths and endpoints shared by the flows. Endpoints come from env vars only (see ADR 0001);
the Makefile exports the host-side values from .env."""

import os
from dataclasses import dataclass
from pathlib import Path

# Model name, aliases, and bucket names live in cxr_core.registry, shared with serving.
TRAIN_EXPERIMENT = "cxr-train"


@dataclass(frozen=True)
class Paths:
    """Repo-relative locations. `root` is the DVC/git repo, found by walking up from cwd."""

    root: Path

    @classmethod
    def discover(cls, start: Path | None = None) -> Paths:
        if env := os.environ.get("CXR_REPO_ROOT"):
            return cls(Path(env).resolve())
        here = (start or Path.cwd()).resolve()
        for candidate in (here, *here.parents):
            if (candidate / ".dvc").is_dir():
                return cls(candidate)
        raise FileNotFoundError(f"no DVC repo (.dvc/) at or above {here}; set CXR_REPO_ROOT")

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def nih(self) -> Path:
        """Train + val images and metadata."""
        return self.data / "nih"

    @property
    def test_locked(self) -> Path:
        """Frozen locked test set: written once by `ingest`, never changed afterwards."""
        return self.data / "test_locked"

    @property
    def versioned(self) -> tuple[Path, Path]:
        """The paths a dataset_rev covers. golden is pinned too, but parity-only."""
        return (self.nih, self.test_locked)

    @property
    def golden(self) -> Path:
        """A few PNG + DICOM files for the E2E parity test, from test_locked patients."""
        return self.data / "golden"

    @property
    def curated(self) -> Path:
        """Reviewer labels from Label Studio (Phase 5). The only place new labels may go."""
        return self.data / "curated"

    @property
    def cache(self) -> Path:
        """Downloads and preprocessed tensors; disposable, never versioned."""
        return self.root / ".cache" / "cxr"


def app_database_url() -> str:
    try:
        return os.environ["APP_DATABASE_URL"]
    except KeyError:
        raise RuntimeError(
            "APP_DATABASE_URL is not set; run through `make`, which builds it from .env"
        ) from None


def require_mlflow_tracking_uri() -> str:
    """Without it MLflow silently uses a local ./mlflow.db, an empty registry that has no
    @champion and isn't the one serving reads."""
    try:
        return os.environ["MLFLOW_TRACKING_URI"]
    except KeyError:
        raise RuntimeError(
            "MLFLOW_TRACKING_URI is not set; run through `make`, which points it at the stack's "
            "MLflow"
        ) from None
