import subprocess
import sys

import cxr_core

SERVING_MODULES = ("cxr_core.preprocess", "cxr_core.cam", "cxr_core.metrics", "cxr_core.bundle")


def test_preprocess_version_is_exported() -> None:
    assert cxr_core.PREPROCESS_VERSION


def test_serving_modules_do_not_import_torch() -> None:
    # A fresh interpreter: other tests in this session import torch on purpose.
    code = (
        "import sys\n"
        + "".join(f"import {m}\n" for m in SERVING_MODULES)
        + "sys.exit('torch' in sys.modules)"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
