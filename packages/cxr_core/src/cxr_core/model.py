"""Model factory: a timm backbone with a GAP + linear head that returns (logits, features).

Needs the `[train]` extra (torch + timm); serving never imports this module and runs the
exported ONNX graph instead. CAM is only exact for GAP + linear heads (see `cam.py`), so
`build_model` rejects any architecture whose own head does more than that: a norm layer
(ConvNeXt), a conv after pooling (MobileNetV3), or token outputs (ViT).
"""

from collections.abc import Callable
from typing import cast

import timm
import torch
from torch import Tensor, nn

# Head-equivalence check input; small enough to be quick for every supported backbone.
_PROBE_SIZE = 64


class UnsupportedArchitectureError(ValueError):
    """The backbone's head is not plain global average pooling + one linear layer."""


class CamModel(nn.Module):
    """Backbone feature map → global average pool → dropout → linear.

    forward() returns `logits` (N, K) and `features` (N, C, h, w), the two ONNX outputs.
    The head is written out here rather than taken from timm, so the exported graph is
    GAP + linear by construction and `head.weight` is exactly what CAM needs.
    """

    def __init__(self, backbone: nn.Module, head: nn.Linear, drop_rate: float = 0.0) -> None:
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(drop_rate)
        self.head = head
        self._forward_features = cast(Callable[[Tensor], Tensor], backbone.forward_features)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        features = self._forward_features(x)
        logits = self.head(self.dropout(self.pool(features)))
        return logits, features

    @staticmethod
    def pool(features: Tensor) -> Tensor:
        """The GAP step, also used to get the embeddings stored in the drift reference."""
        return features.mean(dim=(2, 3))


def build_model(
    architecture: str, num_classes: int, *, pretrained: bool = False, drop_rate: float = 0.0
) -> CamModel:
    """Build a CamModel, or raise UnsupportedArchitectureError if timm's head isn't GAP + linear."""
    net = timm.create_model(architecture, pretrained=pretrained, num_classes=num_classes)
    head = cast(Callable[[], nn.Module], net.get_classifier)()
    if not isinstance(head, nn.Linear):
        raise UnsupportedArchitectureError(
            f"{architecture}: classifier is {type(head).__name__}, expected nn.Linear"
        )
    _check_head_is_gap_linear(architecture, net, head)

    # Drop timm's head; CamModel owns the same linear layer.
    cast(Callable[[int], None], net.reset_classifier)(0)
    return CamModel(net, head, drop_rate=drop_rate)


def _check_head_is_gap_linear(architecture: str, net: nn.Module, head: nn.Linear) -> None:
    """timm's forward_head(features) must equal head(mean(features)) on a random input."""
    net.eval()
    x = torch.randn(2, 3, _PROBE_SIZE, _PROBE_SIZE, generator=torch.Generator().manual_seed(0))
    try:
        with torch.no_grad():
            features = cast(Callable[[Tensor], Tensor], net.forward_features)(x)
            if features.ndim != 4:
                raise UnsupportedArchitectureError(
                    f"{architecture}: forward_features returned shape {tuple(features.shape)}, "
                    "expected a (N, C, h, w) feature map"
                )
            theirs = cast(Callable[[Tensor], Tensor], net.forward_head)(features)
            ours = head(features.mean(dim=(2, 3)))
    except UnsupportedArchitectureError:
        raise
    except Exception as e:  # e.g. a conv between pooling and the classifier changes C
        raise UnsupportedArchitectureError(f"{architecture}: head check failed: {e}") from e
    if theirs.shape != ours.shape or not torch.allclose(theirs, ours, atol=1e-5, rtol=1e-4):
        raise UnsupportedArchitectureError(
            f"{architecture}: head does more than global average pooling + linear"
        )
    net.train()
