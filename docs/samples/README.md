# Sample chest X-rays

Three images to try the UI with, one per finding, so a fresh clone has something real to upload. The dataset itself is never committed (it lives in DVC); these are copies of three `test_locked` images, downscaled from 1024 to 512 px. The model resamples to 128 px anyway.

| File | NIH label (NLP-mined) | Sex, age | View |
| --- | --- | --- | --- |
| `00019087_046_effusion.png` | Effusion | M, 34 | PA |
| `00004342_023_cardiomegaly.png` | Cardiomegaly | M, 44 | PA |
| `00023065_028_pneumothorax.png` | Pneumothorax | F, 50 | PA |

They were chosen as cases the seed model gets right: the labeled finding has the highest probability relative to its threshold. Other findings can still score above their thresholds, which sit low because NIH's prevalence is low (see the [model card](../model-card.md)). They are illustrations, not a benchmark.

**Source:** NIH ChestX-ray14, provided by the NIH Clinical Center, https://nihcc.app.box.com/v/ChestXray-NIHCC. Cite: Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases. IEEE CVPR 2017.
