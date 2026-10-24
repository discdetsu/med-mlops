"""Code shared by training and serving, so both see identical inputs.

`preprocess` and `cam` must stay free of torch: the serving worker imports them
with only numpy and ONNX Runtime installed. Torch lives behind the `[train]` extra.
"""

from cxr_core.preprocess import PREPROCESS_VERSION

__all__ = ["PREPROCESS_VERSION"]
