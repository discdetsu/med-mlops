# 0004: Serve ONNX with two outputs, and allow only GAP + linear heads so CAM is exact

- Status: Accepted
- Date: 2026-10-27

## Context

Serving has to return probabilities and a heatmap from a Celery worker that has no torch (a smaller image, faster start, and no CUDA wheels). Grad-CAM needs gradients, and so it needs the training framework at inference time. Training and serving also have to agree to the decimal on preprocessing and model output, or probabilities and heatmaps go silently wrong.

## Decision

- **Architectures are limited to backbone → global average pooling → one linear layer.** For those, the class activation map `Σ_c W[k, c] · F[c]` is the exact spatial decomposition of logit k (Zhou et al., 2016), so CAM needs only the last feature map and the head weights, with no gradients. `cxr_core.model.build_model` checks this at build time: timm's own `forward_head(features)` must equal `linear(mean(features))` on a probe input. ConvNeXt (norm in the head), MobileNetV3 (conv after pooling), and ViT (tokens, no feature map) fail with `UnsupportedArchitectureError`; ResNet, DenseNet, and EfficientNet pass.
- **`CamModel` writes the head out explicitly** and returns `(logits, features)`, so the exported graph is GAP + linear by construction.
- **ONNX with two outputs and a dynamic batch axis:** `image (N, 3, S, S)` → `logits (N, K)` and `features (N, C, h, w)`. The serving bundle (`cxr_core.bundle`) carries `head.npz` (weight and bias) for CAM, plus the thresholds and preprocess version in `manifest.json`.
- **Shared preprocessing:** `cxr_core.preprocess` (numpy, pydicom, Pillow) builds training tensors from raw files and is the same code the worker runs.
- **E2E parity gates registration.** The train flow runs golden PNG and DICOM files, including MONOCHROME1, through `preprocess` → ONNX Runtime → sigmoid / `cxr_core.cam`, and compares the result with PyTorch on probabilities, features, and every class's CAM. It also checks that the three encodings of one image agree, and that batched and single runs match. Any difference above the tolerance in the config fails the flow before anything is registered. CI runs the same check with a small random model on synthetic golden files.

## Consequences

- The worker needs only numpy, pydicom, Pillow, and onnxruntime.
- Heatmaps are coarse: 4 × 4 for ResNet18 at 128 px, upsampled to the image. The `full` profile (DenseNet121 at 320 px) gives 10 × 10.
- Architectures with attention pooling or head MLPs are ruled out unless this ADR is superseded, for example by an ONNX export of Grad-CAM.
- Export uses torch's TorchScript-based exporter (`dynamo=False`); the dynamo exporter would add onnxscript as a dependency. Parity would catch any difference between the two.

## Alternatives considered

- **Grad-CAM at serving time:** works for any architecture, but needs torch (or a gradient graph exported to ONNX) in the worker and a backward pass per request.
- **Precomputed heatmaps in the ONNX graph:** moves CAM into the graph, so serving can't choose classes or resolution, and parity can't check CAM separately from the features.
- **TorchServe / Triton with PyTorch weights:** heavier to run on a laptop, and training-serving parity would still need its own test.
