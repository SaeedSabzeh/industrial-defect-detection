"""Training loop with deterministic seeding and best-checkpoint restore."""

from __future__ import annotations

import copy
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

__all__ = ["set_seed", "TrainConfig", "History", "train_model"]


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed every RNG that affects a run.

    Without this the original notebooks could not reproduce a number twice,
    which made the differences between architectures impossible to interpret
    -- a 0.5 point gap means nothing if run-to-run variance is unknown.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@dataclass
class TrainConfig:
    model: str = "se_resnet"
    width: int = 64
    reduction: int = 16
    epochs: int = 20
    lr: float = 5e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    image_size: int = 224
    seed: int = 42
    augment: bool = True
    originals_only_train: bool = False
    patience: int = 5
    scheduler_patience: int = 1
    scheduler_factor: float = 0.5
    output_dir: str = "runs"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)

    def append(self, tl: float, ta: float, vl: float, va: float, lr: float) -> None:
        self.train_loss.append(tl)
        self.train_acc.append(ta)
        self.val_loss.append(vl)
        self.val_acc.append(va)
        self.lr.append(lr)


def _run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += outputs.argmax(1).eq(labels).sum().item()
            total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


def train_model(
    model: nn.Module,
    loaders: dict[str, DataLoader],
    cfg: TrainConfig,
    device: torch.device | None = None,
    run_name: str | None = None,
    verbose: bool = True,
) -> tuple[nn.Module, History]:
    """Train, tracking the best validation loss and restoring those weights.

    Early stopping is on by default. The original runs trained a fixed 20-30
    epochs past the best checkpoint, burning compute at learning rates already
    annealed to 1e-6.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    run_name = run_name or f"{cfg.model}_w{cfg.width}_s{cfg.seed}"

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.scheduler_factor, patience=cfg.scheduler_patience
    )

    history = History()
    best_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epochs_without_improvement = 0
    start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = _run_epoch(model, loaders["train"], criterion, optimizer, device, True)
        val_loss, val_acc = _run_epoch(model, loaders["valid"], criterion, optimizer, device, False)

        current_lr = optimizer.param_groups[0]["lr"]
        history.append(train_loss, train_acc, val_loss, val_acc, current_lr)
        scheduler.step(val_loss)

        if verbose:
            print(
                f"[{run_name}] epoch {epoch:3d}/{cfg.epochs} | lr {current_lr:.2e} | "
                f"train {train_loss:.4f}/{train_acc:5.2f}% | val {val_loss:.4f}/{val_acc:5.2f}%"
            )

        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_weights = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if cfg.patience and epochs_without_improvement >= cfg.patience:
                if verbose:
                    print(f"[{run_name}] early stop at epoch {epoch} (best was {best_epoch})")
                break

    elapsed = time.time() - start
    model.load_state_dict(best_weights)

    os.makedirs(cfg.output_dir, exist_ok=True)
    torch.save(best_weights, os.path.join(cfg.output_dir, f"{run_name}.pt"))
    with open(os.path.join(cfg.output_dir, f"{run_name}_history.json"), "w") as fh:
        json.dump(
            {
                "config": asdict(cfg),
                "history": asdict(history),
                "best_val_loss": best_loss,
                "best_epoch": best_epoch,
                "seconds": elapsed,
            },
            fh,
            indent=2,
        )

    if verbose:
        print(f"[{run_name}] done in {elapsed/60:.1f}m | best val loss {best_loss:.4f} @ epoch {best_epoch}")
    return model, history
