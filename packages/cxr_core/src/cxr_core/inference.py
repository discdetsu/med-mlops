"""The ONNX side of inference, shared by the serving worker and the E2E parity test, so parity
checks the exact code serving runs.

No onnxruntime import: callers pass any session with ONNX Runtime's `run` signature.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

ONNX_INPUT = "image"
ONNX_OUTPUTS = ("logits", "features")

FloatArray = npt.NDArray[np.float32]


class OnnxSession(Protocol):
    def run(
        self, output_names: Sequence[str] | None, input_feed: Mapping[str, Any], /
    ) -> Sequence[Any]: ...


def run_onnx(session: OnnxSession, x: FloatArray) -> tuple[FloatArray, FloatArray]:
    """(N, 3, S, S) input → `logits` (N, K) and `features` (N, C, h, w)."""
    logits, features = session.run(list(ONNX_OUTPUTS), {ONNX_INPUT: x})
    return np.asarray(logits, dtype=np.float32), np.asarray(features, dtype=np.float32)


def sigmoid(logits: FloatArray) -> FloatArray:
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
