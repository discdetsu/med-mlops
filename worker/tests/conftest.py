"""A tiny GAP + linear ONNX model built with onnx.helper, so worker tests need no torch."""

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from cxr_core.bundle import Manifest, save_head
from cxr_core.inference import ONNX_INPUT, ONNX_OUTPUTS
from cxr_core.preprocess import PREPROCESS_VERSION

CLASSES = ("Atelectasis", "Effusion")
IMAGE_SIZE = 32
CHANNELS = 4


def build_onnx(path: Path, weight: np.ndarray, bias: np.ndarray, seed: int = 0) -> None:
    """image (N,3,S,S) → Conv 4x4/4 + Relu → features (N,C,S/4,S/4) → GAP → Gemm → logits."""
    conv = np.random.default_rng(seed).normal(0, 0.1, (CHANNELS, 3, 4, 4)).astype(np.float32)
    nodes = [
        helper.make_node(
            "Conv", [ONNX_INPUT, "conv_w"], ["pre"], kernel_shape=[4, 4], strides=[4, 4]
        ),
        helper.make_node("Relu", ["pre"], [ONNX_OUTPUTS[1]]),
        helper.make_node("GlobalAveragePool", [ONNX_OUTPUTS[1]], ["pooled"]),
        helper.make_node("Flatten", ["pooled"], ["flat"]),
        helper.make_node("Gemm", ["flat", "head_w", "head_b"], [ONNX_OUTPUTS[0]], transB=1),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny_cam",
        [
            helper.make_tensor_value_info(
                ONNX_INPUT, TensorProto.FLOAT, ["batch", 3, IMAGE_SIZE, IMAGE_SIZE]
            )
        ],
        [
            helper.make_tensor_value_info(
                ONNX_OUTPUTS[0], TensorProto.FLOAT, ["batch", len(CLASSES)]
            ),
            helper.make_tensor_value_info(
                ONNX_OUTPUTS[1], TensorProto.FLOAT, ["batch", CHANNELS, 8, 8]
            ),
        ],
        initializer=[
            numpy_helper.from_array(conv, "conv_w"),
            numpy_helper.from_array(weight, "head_w"),
            numpy_helper.from_array(bias, "head_b"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


BundleFactory = Callable[..., Path]


@pytest.fixture
def make_bundle(tmp_path: Path) -> BundleFactory:
    def make(
        version: str = "1", preprocess_version: str = PREPROCESS_VERSION, name: str = ""
    ) -> Path:
        bundle = tmp_path / (name or f"bundle-{version}")
        bundle.mkdir()
        rng = np.random.default_rng(int(version) if version.isdigit() else 0)
        weight = rng.normal(0, 1, (len(CLASSES), CHANNELS)).astype(np.float32)
        bias = np.array([-0.5, 0.25], dtype=np.float32)
        build_onnx(bundle / "model.onnx", weight, bias)
        save_head(bundle, weight, bias)
        Manifest(
            model_name="cxr-classifier",
            model_version=version,
            architecture="tiny",
            classes=CLASSES,
            image_size=IMAGE_SIZE,
            preprocess_version=preprocess_version,
            thresholds={"Atelectasis": 0.5, "Effusion": 0.4},
            lineage={"dataset_rev": "rev1", "mlflow_run_id": f"run-{version}"},
        ).save(bundle)
        return bundle

    return make


class FakeS3:
    """download_file / put_object over a dict; bundle prefixes map to local directories."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}

    def add_bundle(self, uri_prefix: str, bundle: Path) -> None:
        bucket, _, prefix = uri_prefix.removeprefix("s3://").partition("/")
        for path in bundle.iterdir():
            self.objects[(bucket, prefix + path.name)] = path.read_bytes()

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        from botocore.exceptions import ClientError

        if (bucket, key) not in self.objects:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        Path(filename).write_bytes(self.objects[(bucket, key)])

    def put_object(self, Bucket: str, Key: str, Body: bytes, **kwargs: Any) -> None:
        self.objects[(Bucket, Key)] = Body


@pytest.fixture
def s3() -> FakeS3:
    return FakeS3()


@pytest.fixture
def golden_png(tmp_path: Path) -> Path:
    from PIL import Image

    path = tmp_path / "image.png"
    gradient = np.linspace(0, 255, 64 * 48, dtype=np.float32).reshape(48, 64)
    Image.fromarray(gradient.astype(np.uint8)).save(path)
    return path


def copy_bundle(src: Path, dest: Path) -> Path:
    shutil.copytree(src, dest)
    return dest
