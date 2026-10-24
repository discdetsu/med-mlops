import importlib
import sys

import cxr_core


def test_preprocess_version_is_exported() -> None:
    assert cxr_core.PREPROCESS_VERSION


def test_serving_modules_do_not_import_torch() -> None:
    for name in ("cxr_core.preprocess", "cxr_core.cam", "cxr_core.metrics"):
        importlib.import_module(name)
    assert "torch" not in sys.modules
