"""Class activation maps from the ONNX `features` output and the linear head weights.

Only valid for GAP + linear head architectures; the model factory enforces that.
With GAP, logit_k = sum_c W[k, c] * mean(F[c]) + b_k, so the map sum_c W[k, c] * F[c]
is exactly the spatial decomposition of that logit (Zhou et al., 2016).
"""

import numpy as np
import numpy.typing as npt

from cxr_core.preprocess import resize


def class_activation_map(
    features: npt.NDArray[np.float32],
    head_weights: npt.NDArray[np.float32],
    class_index: int,
    output_size: tuple[int, int],
) -> npt.NDArray[np.float32]:
    """Return a [0, 1] heatmap of `output_size` (height, width) for one class.

    features: (C, h, w) for one image, the last feature map before pooling.
    head_weights: (num_classes, C), the linear head's weight matrix.
    Negative evidence is clipped to 0; an all-negative map returns all zeros.
    """
    if features.ndim != 3:
        raise ValueError(f"features must be (C, h, w), got shape {features.shape}")
    if head_weights.ndim != 2 or head_weights.shape[1] != features.shape[0]:
        raise ValueError(
            f"head_weights must be (num_classes, {features.shape[0]}), got {head_weights.shape}"
        )
    if not 0 <= class_index < head_weights.shape[0]:
        raise IndexError(f"class_index {class_index} out of range [0, {head_weights.shape[0]})")

    cam = np.tensordot(head_weights[class_index], features, axes=1)
    cam = np.maximum(cam, 0.0).astype(np.float32)
    peak = float(cam.max())
    if peak <= 0.0:
        return np.zeros(output_size, dtype=np.float32)

    heatmap = resize((cam / peak).astype(np.float32), output_size)
    return np.clip(heatmap, 0.0, 1.0)
