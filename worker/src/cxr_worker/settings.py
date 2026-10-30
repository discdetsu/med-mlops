import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    backend_url: str
    internal_token: str
    model_cache: Path
    job_timeout_s: float
    inference_bucket: str = "inference"
    # Long side of the display image and heatmaps written for the UI.
    display_size: int = 512

    @classmethod
    def from_env(cls) -> Settings:
        # CELERY_BROKER_URL is read by app.py at import; checking it here stops a worker
        # from silently falling back to the in-memory broker.
        missing = [
            k
            for k in ("CELERY_BROKER_URL", "BACKEND_URL", "INTERNAL_API_TOKEN")
            if k not in os.environ
        ]
        if missing:
            raise RuntimeError(f"worker needs {', '.join(missing)} (see .env.example)")
        return cls(
            backend_url=os.environ["BACKEND_URL"].rstrip("/"),
            internal_token=os.environ["INTERNAL_API_TOKEN"],
            model_cache=Path(os.environ.get("MODEL_CACHE_DIR", "/tmp/cxr-models")),
            job_timeout_s=float(os.environ.get("JOB_TIMEOUT_SECONDS", "60")),
        )
