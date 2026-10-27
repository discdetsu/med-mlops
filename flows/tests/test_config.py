from pathlib import Path

import pydantic
import pytest

from cxr_flows.config import load_config


@pytest.mark.parametrize("profile", ["demo", "full"])
def test_bundled_profiles_load(profile: str) -> None:
    cfg = load_config(profile)
    assert cfg.profile == profile
    assert len(cfg.config_hash) == 12


def test_demo_is_the_cpu_profile() -> None:
    cfg = load_config("demo")
    assert (cfg.architecture, cfg.image_size) == ("resnet18", 128)


def test_unknown_keys_and_bad_values_are_rejected(tmp_path: Path) -> None:
    base = (Path(__file__).parents[1] / "src/cxr_flows/configs/demo.yaml").read_text()
    typo = tmp_path / "typo.yaml"
    typo.write_text(base + "learning_rate: 0.1\n")
    with pytest.raises(pydantic.ValidationError):
        load_config(str(typo))
    bad = tmp_path / "bad.yaml"
    bad.write_text(base.replace("image_size: 128", "image_size: 8"))
    with pytest.raises(pydantic.ValidationError):
        load_config(str(bad))


def test_unknown_profile() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("nope")


def test_hash_changes_with_config() -> None:
    cfg = load_config("demo")
    assert cfg.config_hash != cfg.model_copy(update={"epochs": cfg.epochs + 1}).config_hash
