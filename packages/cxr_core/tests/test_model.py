import pytest
import torch

from cxr_core.model import CamModel, UnsupportedArchitectureError, build_model


@pytest.mark.parametrize("architecture", ["resnet18", "efficientnet_b0"])
def test_gap_linear_architectures_build(architecture: str) -> None:
    model = build_model(architecture, num_classes=5).eval()
    with torch.no_grad():
        logits, features = model(torch.randn(2, 3, 64, 64))
    assert logits.shape == (2, 5)
    assert features.ndim == 4 and features.shape[0] == 2
    assert model.head.weight.shape == (5, features.shape[1])


def test_logits_are_the_linear_head_of_the_pooled_features() -> None:
    """The property CAM relies on: logits = W · mean(features) + b."""
    model = build_model("resnet18", num_classes=3).eval()
    with torch.no_grad():
        logits, features = model(torch.randn(1, 3, 64, 64))
        expected = model.head(CamModel.pool(features))
    torch.testing.assert_close(logits, expected)


@pytest.mark.parametrize(
    "architecture",
    [
        "convnext_atto",  # norm layer in the head
        "mobilenetv3_small_050",  # conv between pooling and classifier
        "vit_tiny_patch16_224",  # token outputs, no spatial map
    ],
)
def test_other_heads_are_rejected_at_build_time(architecture: str) -> None:
    with pytest.raises(UnsupportedArchitectureError):
        build_model(architecture, num_classes=5)
