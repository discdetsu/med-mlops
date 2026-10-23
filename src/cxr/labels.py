"""The 14 NIH ChestX-ray14 findings.

The order here is the model's output order (logit index) and the ids of
`dataset.findings`. Never reorder; only append.
"""

CLASSES: tuple[str, ...] = (
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
)

# Appears in `Finding Labels` when an image has none of the classes above.
NO_FINDING = "No Finding"
