"""CPU training loop for a CamModel, keeping the epoch with the best validation macro AUROC."""

import copy
import logging
import math
import os
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from cxr_core.inference import sigmoid
from cxr_core.metrics import per_class_auroc
from cxr_core.model import CamModel, build_model
from cxr_core.preprocess import IMAGENET_MEAN, IMAGENET_STD
from cxr_flows.config import TrainConfig
from cxr_flows.dataset import Split

log = logging.getLogger(__name__)


@dataclass
class TrainResult:
    model: CamModel
    best_epoch: int
    history: list[dict[str, float]] = field(default_factory=list)


@dataclass(frozen=True)
class Predictions:
    logits: npt.NDArray[np.float32]  # (N, K)
    embeddings: npt.NDArray[np.float32]  # (N, C), the pooled features

    @property
    def probabilities(self) -> npt.NDArray[np.float32]:
        return sigmoid(self.logits)


class _Inputs(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, split: Split, transform: Callable[[Tensor], Tensor] | None) -> None:
        self.split = split
        self.transform = transform

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        x = torch.from_numpy(np.array(self.split.x[i]))  # copy out of the memmap
        if self.transform is not None:
            x = self.transform(x)
        return x, torch.tensor(self.split.y[i])


def augmentation() -> Callable[[Tensor], Tensor]:
    """Small geometric jitter on the normalized tensor. No flips: the heart is on the left.

    Pixels moved in from outside are filled with what black (0.0) becomes after normalization.
    """
    black = (-IMAGENET_MEAN / IMAGENET_STD).tolist()
    transform: Callable[[Tensor], Tensor] = v2.RandomAffine(
        degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05), fill=black
    )
    return transform


def train_model(cfg: TrainConfig, train: Split, val: Split, num_classes: int) -> TrainResult:
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.num_threads or os.cpu_count() or 1)

    model = build_model(
        cfg.architecture, num_classes, pretrained=cfg.pretrained, drop_rate=cfg.drop_rate
    )
    loader = DataLoader(
        _Inputs(train, augmentation() if cfg.augment else None),
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr, total_steps=cfg.epochs * len(loader), pct_start=0.15
    )
    loss_fn = nn.BCEWithLogitsLoss()

    result = TrainResult(model=model, best_epoch=0)
    best_auroc, best_state = -math.inf, copy.deepcopy(model.state_dict())
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for x, y in loader:
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.item() * len(x)
            seen += len(x)

        preds = predict(model, val.x, cfg.batch_size * 2)
        val_loss = float(loss_fn(torch.from_numpy(preds.logits), torch.from_numpy(val.y)))
        val_auroc = float(np.nanmean(per_class_auroc(val.y, preds.probabilities)))
        metrics = {"train_loss": total / seen, "val_loss": val_loss, "val_macro_auroc": val_auroc}
        result.history.append({"epoch": epoch, **metrics})
        log.info(
            "epoch %d/%d  train_loss %.4f  val_loss %.4f  val_macro_auroc %.4f",
            epoch,
            cfg.epochs,
            metrics["train_loss"],
            val_loss,
            val_auroc,
        )
        if val_auroc > best_auroc:
            best_auroc, best_state = val_auroc, copy.deepcopy(model.state_dict())
            result.best_epoch = epoch

    model.load_state_dict(best_state)
    model.eval()
    return result


def predict(model: CamModel, x: npt.NDArray[np.float32], batch_size: int) -> Predictions:
    model.eval()
    logits, embeddings = [], []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(
                np.array(x[start : start + batch_size])
            )  # copy: memmap is read-only
            out, features = model(batch)
            logits.append(out.numpy())
            embeddings.append(CamModel.pool(features).numpy())
    return Predictions(
        logits=np.concatenate(logits).astype(np.float32),
        embeddings=np.concatenate(embeddings).astype(np.float32),
    )
