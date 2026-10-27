# Model card: cxr-classifier

> **Not a medical device.** This is a portfolio project. It is not clinically validated and must not be used for diagnosis.

## Model

- **Task:** multi-label classification of 5 findings on frontal chest X-rays, with a class activation map per finding.
- **Architecture:** a timm backbone → global average pooling → one linear layer, the only head shape the factory accepts, so CAM is exact ([ADR 0004](adr/0004-onnx-gap-linear-cam.md)). `demo` profile: ResNet18, ImageNet-pretrained, 128 px. `full` profile: DenseNet121, 320 px.
- **Serving format:** ONNX with `logits` and `features` outputs, plus the head weights, per-class thresholds, and drift reference in one bundle.
- **Lineage:** every registered version carries `dataset_rev`, `git_commit`, `preprocess_version`, `prefect_flow_run_id`, `config_hash`, and its thresholds as MLflow tags.

## Data

[NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC): 112,120 frontal PNGs (1024 × 1024) of 30,805 patients, de-identified by NIH, with 14 findings mined from radiology reports by NLP. This project uses a subset downloaded from the public Kaggle mirror:

| Split | Images | Patients | Atelectasis | Cardiomegaly | Effusion | Mass | Pneumothorax | AP views | Role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 2,822 | 2,035 | 592 | 385 | 734 | 474 | 444 | 1,196 | training |
| val | 579 | 432 | 108 | 88 | 151 | 87 | 111 | 228 | epoch selection, thresholds, drift reference |
| test_locked | 599 | 462 | 130 | 87 | 142 | 85 | 66 | 226 | reported metrics and the promotion gate; frozen |

The subset is sampled with a fixed seed: up to 500 images per class, plus 1,500 images with none of the 5 findings (`No Finding` or other findings only). The split is a hash of the patient ID, 70/15/15, so no patient is in more than one split ([ADR 0003](adr/0003-dataset-versions-and-locked-test.md)). The negative-to-positive ratio is therefore much lower than in NIH overall, so metrics that depend on prevalence (ECE, positive predictive value) don't transfer to the full dataset or to clinical prevalence.

## Classes

| Class | NIH images | Why it is in |
| --- | --- | --- |
| Atelectasis | 11,559 | common; frequent in follow-up and ICU (AP) studies |
| Cardiomegaly | 2,776 | a global finding (heart size), visible at low resolution, and the clearest CAM demo |
| Effusion | 13,317 | common, and among the best-learned NIH labels in published baselines |
| Mass | 5,782 | a focal finding, so the CAM has to localize |
| Pneumothorax | 5,302 | urgent finding; a known shortcut risk (chest drains), worth showing in subgroup results |

Left out:

- **Infiltration** (the most common finding): the noisiest NIH label, overlapping with several others.
- **Nodule:** usually a few pixels at 128 px.
- **Consolidation, Pneumonia, Edema:** heavily overlap with Infiltration and Effusion in the NLP labels.
- **Hernia, Fibrosis, Emphysema, Pleural thickening:** too few examples for a small subset, or too subtle at demo resolution.

## Metrics

From the `@champion` version's `reports/evaluation.json` in MLflow. Thresholds are chosen on val (maximum Youden's J); metrics are on test_locked (599 images, 462 patients), with 95% bootstrap CIs over images (500 resamples).

**Version 1** (`demo` profile: ResNet18, 128 px, 6 epochs on CPU, `dataset_rev` `bbc133af190d`): macro AUROC **0.700** (0.674 to 0.724), pooled ECE 0.026.

| Class | Positives | AUROC (95% CI) | Sens @ spec 0.9 | Threshold | Sens | Spec | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Atelectasis | 130 | 0.722 (0.676 to 0.769) | 0.277 | 0.165 | 0.808 | 0.478 | 0.045 |
| Cardiomegaly | 87 | 0.661 (0.607 to 0.720) | 0.241 | 0.135 | 0.678 | 0.518 | 0.018 |
| Effusion | 142 | 0.784 (0.743 to 0.823) | 0.408 | 0.242 | 0.796 | 0.608 | 0.028 |
| Mass | 85 | 0.557 (0.491 to 0.621) | 0.153 | 0.179 | 0.459 | 0.640 | 0.049 |
| Pneumothorax | 66 | 0.775 (0.709 to 0.834) | 0.470 | 0.148 | 0.788 | 0.608 | 0.052 |

| Subgroup | n | Macro AUROC |
| --- | --- | --- |
| Male | 328 | 0.705 |
| Female | 271 | 0.698 |
| PA view | 373 | 0.722 |
| AP view | 226 | 0.646 |
| Age < 40 | 172 | 0.740 |
| Age 40 to 59 | 288 | 0.688 |
| Age 60+ | 139 | 0.671 |

This version proves the pipeline, not the model. On the full dataset at 224 px or more, published baselines reach about 0.71 to 0.81 on these five classes (Wang et al., 2017) and 0.81 to 0.93 (CheXNet, Rajpurkar et al., 2017), on NIH's official split rather than this one. Validation AUROC was still rising at the last epoch, and 128 px is coarse, Cardiomegaly especially. A longer run, the `full` profile, or a bigger subset should close most of the gap. The weaker AP subgroup matches the shortcut risk below.

## Limitations

- **Noisy labels:** NIH labels were mined from reports by NLP, with reported accuracy above 90% overall but much lower for some findings. Some "negatives" are positives and vice versa, which caps achievable AUROC and makes small differences between models meaningless without the CIs.
- **Small test set:** a few hundred images give wide CIs per class and wider ones per subgroup; subgroups below 30 images are not scored.
- **Bootstrap over images, not patients:** patients with several test images make the CIs somewhat too narrow.
- **One institution:** all images come from the NIH Clinical Center, so performance on other scanners, protocols, or populations is unknown, which is what the Phase 4 drift monitoring is for.
- **Shortcuts:** AP views are mostly inpatient and sicker patients, and pneumothorax images often show chest drains. The model can learn these instead of the finding; the view subgroup results are the first place to look.
- **Low resolution in `demo`:** at 128 px, small findings are lost and the CAM is a 4 × 4 grid upsampled to the image.
