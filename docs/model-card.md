# Model card: cxr-classifier

> **Not a medical device.** This is a portfolio project. It is not clinically validated and must not be used for diagnosis.

## Model

- **Task:** multi-label classification of 5 findings on frontal chest X-rays, with a class activation map per finding.
- **Architecture:** a timm backbone → global average pooling → one linear layer, the only head shape the factory accepts, so CAM is exact ([ADR 0004](adr/0004-onnx-gap-linear-cam.md)). `demo` profile: ResNet18, ImageNet-pretrained, 128 px. `full` profile: DenseNet121, 320 px.
- **Serving format:** ONNX with `logits` and `features` outputs, plus the head weights, per-class thresholds, and drift reference in one bundle.
- **Lineage:** every registered version carries `dataset_rev`, `git_commit`, `preprocess_version`, `prefect_flow_run_id`, `config_hash`, and its thresholds as MLflow tags.

## Data

[NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC): 112,120 frontal PNGs (1024 × 1024) of 30,805 patients, de-identified by NIH, with 14 findings mined from radiology reports by NLP. The split is a hash of the patient ID, 70/15/15, so no patient is in more than one split ([ADR 0003](adr/0003-dataset-versions-and-locked-test.md)).

**Current dataset, `dataset_rev` `f774f7a855ab`:** every NIH image, at NIH's own prevalence.

| Split | Images | Patients | Atelectasis | Cardiomegaly | Effusion | Mass | Pneumothorax | AP views | Role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 79,339 | 21,489 | 8,224 | 1,986 | 9,542 | 4,131 | 3,844 | 32,327 | training |
| val | 16,166 | 4,646 | 1,626 | 424 | 1,802 | 800 | 737 | 5,999 | epoch selection, thresholds, drift reference |
| test_locked | 16,615 | 4,670 | 1,709 | 366 | 1,973 | 851 | 721 | 6,484 | reported metrics and the promotion gate; frozen |

**Version 1's dataset, `dataset_rev` `bbc133af190d`:** the default `make seed` subset. Up to 500 images per class plus 1,500 with none of the 5 findings (`No Finding` or other findings only), sampled with a fixed seed. Its negative-to-positive ratio is much lower than NIH's, so its prevalence-dependent metrics (ECE, positive predictive value) don't transfer to the full dataset or to clinical prevalence.

| Split | Images | Patients | Atelectasis | Cardiomegaly | Effusion | Mass | Pneumothorax | AP views |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 2,822 | 2,035 | 592 | 385 | 734 | 474 | 444 | 1,196 |
| val | 579 | 432 | 108 | 88 | 151 | 87 | 111 | 228 |
| test_locked | 599 | 462 | 130 | 87 | 142 | 85 | 66 | 226 |

The locked test set was rebuilt once, with the move to the full dataset, while version 1 was the only model. Every model trained from now on is evaluated on the 16,615-image set.

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

From the `@champion` version's `reports/evaluation.json` in MLflow. Thresholds are chosen on val (maximum Youden's J); metrics have 95% bootstrap CIs over images (500 resamples).

**Version 2, `@champion`** (`demo` profile: ResNet18, 128 px, 6 epochs on CPU, `dataset_rev` `f774f7a855ab`). Trained on a fixed 4,000-image sample of the train split; thresholds and epoch selection use the whole val split. On the 16,615-image test set (4,670 patients): macro AUROC **0.708** (0.700 to 0.717), pooled ECE 0.006. ECE and the thresholds are low because the test set has NIH's own prevalence.

| Class | Positives | AUROC (95% CI) | Sens @ spec 0.9 | Threshold | Sens | Spec | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Atelectasis | 1,709 | 0.715 (0.704 to 0.726) | 0.264 | 0.092 | 0.708 | 0.608 | 0.009 |
| Cardiomegaly | 366 | 0.636 (0.608 to 0.666) | 0.150 | 0.018 | 0.656 | 0.572 | 0.004 |
| Effusion | 1,973 | 0.799 (0.790 to 0.808) | 0.429 | 0.084 | 0.811 | 0.646 | 0.018 |
| Mass | 851 | 0.644 (0.624 to 0.664) | 0.208 | 0.042 | 0.632 | 0.567 | 0.008 |
| Pneumothorax | 721 | 0.748 (0.730 to 0.764) | 0.359 | 0.039 | 0.707 | 0.659 | 0.004 |

| Subgroup | n | Macro AUROC |
| --- | --- | --- |
| Male | 9,279 | 0.702 |
| Female | 7,336 | 0.715 |
| PA view | 10,131 | 0.740 |
| AP view | 6,484 | 0.642 |
| Age < 40 | 5,532 | 0.718 |
| Age 40 to 59 | 7,112 | 0.712 |
| Age 60+ | 3,965 | 0.674 |

This version proves the pipeline, not the model. On the full dataset at 224 px or more, published baselines reach about 0.71 to 0.81 on these five classes (Wang et al., 2017) and 0.81 to 0.93 (CheXNet, Rajpurkar et al., 2017), on NIH's official split rather than this one. It sees 5% of the training images, and 128 px is coarse, Cardiomegaly especially; validation AUROC levelled off at about 0.71 from epoch 4. More training images, the `full` profile, or both should close most of the gap. The AP gap (0.642 vs 0.740 for PA) matches the shortcut risk below.

**Version 1** (same profile, `dataset_rev` `bbc133af190d`, no longer champion). Its numbers are on the *old* 599-image test set (462 patients) and can't be compared with version 2 or anything later: macro AUROC **0.700** (0.674 to 0.724), pooled ECE 0.026.

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

## Limitations

- **Noisy labels:** NIH labels were mined from reports by NLP, with reported accuracy above 90% overall but much lower for some findings. Some "negatives" are positives and vice versa, which caps achievable AUROC and makes small differences between models meaningless without the CIs.
- **Small test set (version 1):** 599 images give wide CIs per class and wider ones per subgroup; subgroups below 30 images are not scored. The current 16,615-image test set fixes this for later versions.
- **Bootstrap over images, not patients:** patients with several test images make the CIs somewhat too narrow.
- **One institution:** all images come from the NIH Clinical Center, so performance on other scanners, protocols, or populations is unknown, which is what the Phase 4 drift monitoring is for.
- **Shortcuts:** AP views are mostly inpatient and sicker patients, and pneumothorax images often show chest drains. The model can learn these instead of the finding; the view subgroup results are the first place to look.
- **Low resolution in `demo`:** at 128 px, small findings are lost and the CAM is a 4 × 4 grid upsampled to the image.
