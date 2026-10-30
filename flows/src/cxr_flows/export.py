"""ONNX export and the E2E parity test.

Parity starts from raw golden files (PNG and DICOM, including MONOCHROME1) and runs
`cxr_core.preprocess` → ONNX Runtime → sigmoid / `cxr_core.cam`, the path serving takes,
then compares it with PyTorch on the same inputs. It also checks that the PNG and both
DICOM encodings of one image give the same probabilities through ONNX.
"""

import warnings
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnx
import onnxruntime as ort
import torch

from cxr_core.cam import class_activation_map
from cxr_core.inference import ONNX_INPUT, ONNX_OUTPUTS, run_onnx, sigmoid
from cxr_core.model import CamModel
from cxr_core.preprocess import load_image, preprocess
from cxr_flows.config import ParityTolerance

OPSET = 18


def export_onnx(model: CamModel, image_size: int, path: Path) -> Path:
    """Export with a dynamic batch axis on the input and both outputs."""
    model.eval()
    dummy = torch.zeros(1, 3, image_size, image_size)
    with warnings.catch_warnings():
        # The TorchScript exporter is deprecated in favour of dynamo, which needs onnxscript;
        # for a plain CNN its output is identical, and parity below checks that anyway.
        warnings.simplefilter("ignore", category=DeprecationWarning)
        torch.onnx.export(
            model,
            (dummy,),
            str(path),
            input_names=[ONNX_INPUT],
            output_names=list(ONNX_OUTPUTS),
            dynamic_axes={name: {0: "batch"} for name in (ONNX_INPUT, *ONNX_OUTPUTS)},
            opset_version=OPSET,
            dynamo=False,
        )
    onnx.checker.check_model(onnx.load(str(path)))
    return path


@dataclass
class FileParity:
    file: str
    image_id: str
    probabilities: float
    features: float
    cam: float


@dataclass
class ParityReport:
    passed: bool
    tolerance: dict[str, float]
    files: list[FileParity] = field(default_factory=list)
    cross_format: dict[str, float] = field(default_factory=dict)  # image_id → max prob diff
    batch: float = 0.0  # batched vs single ONNX run, max prob diff
    failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_parity(
    model: CamModel,
    onnx_path: Path,
    golden: Sequence[tuple[Path, str]],
    image_size: int,
    tolerance: ParityTolerance,
) -> ParityReport:
    """`golden` is (file, image_id) pairs; files with the same image_id are re-encodings."""
    if not golden:
        raise ValueError("parity needs at least one golden file")
    model.eval()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    weight = model.head.weight.detach().numpy().astype(np.float32)
    size = (image_size, image_size)
    report = ParityReport(passed=False, tolerance=tolerance.model_dump())

    onnx_probs: dict[str, list[npt.NDArray[np.float32]]] = {}
    inputs = []
    for path, image_id in golden:
        x = preprocess(load_image(path), image_size)
        inputs.append(x)
        with torch.inference_mode():
            t_logits, t_features = (t.numpy() for t in model(torch.from_numpy(x)))
        o_logits, o_features = run_onnx(session, x)

        t_probs, o_probs = sigmoid(t_logits), sigmoid(o_logits)
        cam_diff = max(
            float(
                np.abs(
                    class_activation_map(t_features[0], weight, k, size)
                    - class_activation_map(o_features[0], weight, k, size)
                ).max()
            )
            for k in range(weight.shape[0])
        )
        report.files.append(
            FileParity(
                file=path.name,
                image_id=image_id,
                probabilities=float(np.abs(t_probs - o_probs).max()),
                features=float(np.abs(t_features - o_features).max()),
                cam=cam_diff,
            )
        )
        onnx_probs.setdefault(image_id, []).append(o_probs[0])

    for image_id, probs in onnx_probs.items():
        stacked = np.stack(probs)
        report.cross_format[image_id] = float((stacked.max(axis=0) - stacked.min(axis=0)).max())

    batched_logits, _ = run_onnx(session, np.concatenate(inputs))
    single = np.concatenate([run_onnx(session, x)[0] for x in inputs])
    report.batch = float(np.abs(sigmoid(batched_logits) - sigmoid(single)).max())

    for f in report.files:
        for metric in ("probabilities", "features", "cam"):
            if getattr(f, metric) > getattr(tolerance, metric):
                report.failures.append(f"{f.file}: {metric} diff {getattr(f, metric):.2e}")
    for image_id, diff in report.cross_format.items():
        if diff > tolerance.cross_format:
            report.failures.append(f"{image_id}: PNG vs DICOM probability diff {diff:.2e}")
    if report.batch > tolerance.probabilities:
        report.failures.append(f"batched vs single probability diff {report.batch:.2e}")
    report.passed = not report.failures
    return report
