"""Dataset indexing and leakage-free splitting.

Background
----------
The Industry Biscuit dataset ships 4,900 images that are *not* 4,900
independent samples. They are 1,225 base images, each followed by three
augmented variants -- 1,225 x 4 = 4,900 exactly.

The original project's split walked the annotation CSV in order and filled
train, then valid, then test. Because augmented variants of the same base
image sit at predictable offsets in that CSV, variants of one physical
biscuit could land on both sides of a split boundary. A model can then score
near-perfectly on the test set by recognising an image it already memorised
in a slightly rotated form.

This module fixes that by splitting at the *group* level: all four variants
of a base image always travel together.

CSV layout (1-indexed rows), reconstructed from the original loader:
    group k  ->  row k                       (base image,  k = 1..1225)
                 row 1226 + 3*(k-1)          (variant 1)
                 row 1227 + 3*(k-1)          (variant 2)
                 row 1228 + 3*(k-1)          (variant 3)

`verify_grouping` checks this assumption against the labels and raises if it
does not hold, so a change in dataset layout fails loudly rather than
silently reintroducing leakage.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

__all__ = [
    "N_BASE_IMAGES",
    "OK_CLASS",
    "assign_group_ids",
    "build_index",
    "verify_grouping",
    "group_split",
    "BiscuitDataset",
    "build_transforms",
    "make_loaders",
    "SplitSizes",
]

N_BASE_IMAGES = 1225
OK_CLASS = "Defect_No"

# ImageNet statistics; kept for comparability with the original notebooks.
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def assign_group_ids(n_rows: int, n_base: int = N_BASE_IMAGES) -> np.ndarray:
    """Map 0-indexed CSV row positions to base-image group ids.

    Rows [0, n_base) are base images. Rows [n_base, n_rows) are variants,
    three per base image, in base-image order.
    """
    if n_rows < n_base:
        raise ValueError(f"expected at least {n_base} rows, got {n_rows}")

    groups = np.empty(n_rows, dtype=np.int64)
    groups[:n_base] = np.arange(n_base)

    n_variants = n_rows - n_base
    if n_variants:
        # variant i belongs to base image i // 3
        groups[n_base:] = np.arange(n_variants) // 3

    if groups.max() >= n_base:
        raise ValueError(
            f"variant rows imply {groups.max() + 1} groups but only {n_base} base "
            "images were declared -- dataset layout differs from expectation"
        )
    return groups


def build_index(csv_path: str, n_base: int = N_BASE_IMAGES) -> pd.DataFrame:
    """Read the annotation CSV into a tidy frame with group ids and binary labels."""
    df = pd.read_csv(csv_path, usecols=["file", "classDescription"])
    df = df.reset_index(drop=True)

    df["group_id"] = assign_group_ids(len(df), n_base=n_base)
    df["is_original"] = np.arange(len(df)) < n_base
    df["defect_type"] = df["classDescription"]
    # label 1 = defective (nok). The positive class is the defect, because
    # defect recall is the metric that matters on a production line.
    df["label"] = (df["classDescription"] != OK_CLASS).astype(int)
    return df[["file", "defect_type", "label", "group_id", "is_original"]]


def verify_grouping(df: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    """Check that every group carries a single consistent label.

    An augmented variant must share its base image's class. If groups mix
    labels, the offset assumption in `assign_group_ids` is wrong for this
    dataset version and the split would not actually prevent leakage.
    """
    per_group = df.groupby("group_id")["label"].nunique()
    inconsistent = per_group[per_group > 1]

    sizes = df.groupby("group_id").size()
    odd_sizes = sizes[sizes != 4]

    report = pd.DataFrame(
        {
            "n_groups": [len(per_group)],
            "n_inconsistent_labels": [len(inconsistent)],
            "n_groups_not_size_4": [len(odd_sizes)],
        }
    )

    if strict and len(inconsistent):
        raise ValueError(
            f"{len(inconsistent)} groups contain more than one label. The "
            "base/variant offset assumption does not hold for this dataset "
            "version -- inspect the CSV before training, or the split will "
            "leak. Pass strict=False to override."
        )
    return report


@dataclass
class SplitSizes:
    train: int
    valid: int
    test: int

    def as_dict(self) -> dict[str, int]:
        return {"train": self.train, "valid": self.valid, "test": self.test}


def group_split(
    df: pd.DataFrame,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> dict[str, pd.DataFrame]:
    """Split by group, stratified on label, with a fixed seed.

    Stratification is done on the group's label (constant within a group by
    construction), so class balance is preserved across splits without ever
    separating a base image from its variants.
    """
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"ratios must sum to 1.0, got {sum(ratios)}")

    rng = np.random.default_rng(seed)
    group_labels = df.groupby("group_id")["label"].first()

    assignment: dict[int, str] = {}
    for label_value in sorted(group_labels.unique()):
        groups = group_labels[group_labels == label_value].index.to_numpy()
        groups = rng.permutation(groups)

        n = len(groups)
        n_train = int(round(n * ratios[0]))
        n_valid = int(round(n * ratios[1]))

        for g in groups[:n_train]:
            assignment[g] = "train"
        for g in groups[n_train : n_train + n_valid]:
            assignment[g] = "valid"
        for g in groups[n_train + n_valid :]:
            assignment[g] = "test"

    df = df.copy()
    df["split"] = df["group_id"].map(assignment)
    splits = {name: part.reset_index(drop=True) for name, part in df.groupby("split")}

    # Hard guarantee: no group id may appear in more than one split.
    seen: dict[int, str] = {}
    for name, part in splits.items():
        for g in part["group_id"].unique():
            if g in seen:
                raise AssertionError(f"group {g} appears in both {seen[g]} and {name}")
            seen[g] = name

    return splits


def build_transforms(train: bool, image_size: int = 224, augment: bool = True):
    """Train-time augmentation is applied only when `train and augment`.

    The original pipeline used resize + normalise everywhere and relied on the
    dataset's pre-baked variants for augmentation -- which is what made the
    leakage possible. Live augmentation on the train split only is the safer
    equivalent.
    """
    steps = [transforms.Resize((image_size, image_size))]
    if train and augment:
        steps += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        ]
    steps += [transforms.ToTensor(), transforms.Normalize(mean=_MEAN, std=_STD)]
    return transforms.Compose(steps)


class BiscuitDataset(Dataset):
    """Reads images straight from the source directory -- no file copying.

    The original pipeline wrote 4,900 re-encoded JPEGs into a scratch folder
    on every run. Indexing the source directory instead removes several
    minutes of setup and one generation of JPEG recompression.
    """

    classes = ["ok", "nok"]

    def __init__(self, frame: pd.DataFrame, images_dir: str, transform=None):
        self.frame = frame.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        path = os.path.join(self.images_dir, row["file"])
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["label"])


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


def make_loaders(
    splits: dict[str, pd.DataFrame],
    images_dir: str,
    batch_size: int = 32,
    image_size: int = 224,
    augment: bool = True,
    num_workers: int = 2,
    seed: int = 42,
    originals_only_train: bool = False,
) -> dict[str, DataLoader]:
    """Build train/valid/test loaders with deterministic shuffling.

    `originals_only_train` drops the dataset's pre-baked variants from the
    training split and relies on live augmentation instead. Useful for
    measuring how much the shipped variants actually contribute.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)

    loaders: dict[str, DataLoader] = {}
    for name, frame in splits.items():
        is_train = name == "train"
        if is_train and originals_only_train:
            frame = frame[frame["is_original"]].reset_index(drop=True)

        dataset = BiscuitDataset(
            frame,
            images_dir,
            transform=build_transforms(train=is_train, image_size=image_size, augment=augment),
        )
        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            worker_init_fn=_seed_worker,
            generator=generator if is_train else None,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders
