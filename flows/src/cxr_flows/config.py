"""Training config profiles (`configs/<profile>.yaml`), validated with pydantic.

A config is frozen once loaded; its hash goes on the MLflow run so two runs with the same
dataset_rev and config_hash are directly comparable.
"""

import hashlib
from importlib import resources
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ParityTolerance(BaseModel):
    """Max absolute ONNX-vs-PyTorch difference allowed on the golden set."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    probabilities: float = Field(default=1e-4, gt=0)
    features: float = Field(default=1e-3, gt=0)
    cam: float = Field(default=1e-3, gt=0)
    # PNG vs MONOCHROME1/2 DICOM of the same image, all through ONNX.
    cross_format: float = Field(default=1e-4, gt=0)


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    profile: str
    architecture: str
    pretrained: bool = True
    image_size: int = Field(ge=32, le=1024)
    epochs: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    lr: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    drop_rate: float = Field(default=0.0, ge=0, lt=1)
    augment: bool = True
    max_train_images: int | None = Field(default=None, ge=1)
    seed: int = 0
    num_threads: int | None = Field(default=None, ge=1)  # torch threads; None = all cores
    # Report sensitivity at this specificity, a common operating point for screening.
    target_specificity: float = Field(default=0.9, gt=0, lt=1)
    bootstrap_samples: int = Field(default=500, ge=0)
    min_subgroup_size: int = Field(default=30, ge=1)
    parity: ParityTolerance = ParityTolerance()

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:12]


def load_config(profile_or_path: str) -> TrainConfig:
    """Load a bundled profile by name (`demo`, `full`) or any YAML file by path."""
    if profile_or_path.endswith((".yaml", ".yml")):
        text = Path(profile_or_path).read_text()
    else:
        ref = resources.files("cxr_flows") / "configs" / f"{profile_or_path}.yaml"
        if not ref.is_file():
            raise FileNotFoundError(f"no config profile {profile_or_path!r} in cxr_flows/configs")
        text = ref.read_text()
    return TrainConfig.model_validate(yaml.safe_load(text))
