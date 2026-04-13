"""
dataset.py – PyTorch Dataset, DataLoader, and class-balancing utilities.

Provides:
  - AntiSpoofDataset          (torch Dataset wrapper)
  - get_class_weights          (inverse-frequency weights for loss)
  - get_weighted_sampler       (WeightedRandomSampler for oversampling)
  - create_dataloaders         (factory returning train/val/test loaders)
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

from . import config
from .augmentation import get_eval_transforms, get_train_transforms

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────


class AntiSpoofDataset(Dataset):
    """Thin wrapper around ``ImageFolder`` that lets callers swap transforms
    and query class counts easily."""

    def __init__(self, root: str | Path, transform: Optional[transforms.Compose] = None):
        self.root = Path(root)
        self._dataset = ImageFolder(str(self.root), transform=transform)
        self.classes = self._dataset.classes
        self.class_to_idx = self._dataset.class_to_idx
        self.targets = self._dataset.targets
        self.samples = self._dataset.samples

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int):
        return self._dataset[index]

    # ── convenience helpers ────────────────────────────────────

    def class_counts(self) -> Dict[str, int]:
        """Return  { class_name: count }."""
        counts: Dict[str, int] = {}
        for cls_name in self.classes:
            idx = self.class_to_idx[cls_name]
            counts[cls_name] = self.targets.count(idx)
        return counts

    def print_distribution(self, split_name: str = "") -> None:
        counts = self.class_counts()
        header = f"[{split_name}] " if split_name else ""
        total = sum(counts.values())
        logger.info("%sClass distribution (total=%d):", header, total)
        for cls_name, c in counts.items():
            pct = 100.0 * c / total if total else 0
            logger.info("  %-8s : %5d  (%.1f%%)", cls_name, c, pct)


# ──────────────────────────────────────────────────────────────
# Class weighting / oversampling
# ──────────────────────────────────────────────────────────────


def get_class_weights(dataset: AntiSpoofDataset) -> torch.Tensor:
    """Compute inverse-frequency class weights for ``CrossEntropyLoss``.

    Returns a float tensor of shape ``(num_classes,)``.
    """
    counts = dataset.class_counts()
    total = sum(counts.values())
    num_classes = len(counts)
    weights = []
    for cls_name in dataset.classes:
        c = counts[cls_name]
        w = total / (num_classes * c) if c > 0 else 1.0
        weights.append(w)

    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    logger.info("Class weights: %s", dict(zip(dataset.classes, weights)))
    return weight_tensor


def get_weighted_sampler(dataset: AntiSpoofDataset) -> WeightedRandomSampler:
    """Build a ``WeightedRandomSampler`` that oversamples the minority class
    so that each epoch sees a balanced distribution."""
    counts = dataset.class_counts()
    total = sum(counts.values())
    class_weights = {
        cls_name: total / c if c > 0 else 1.0
        for cls_name, c in counts.items()
    }

    sample_weights = [
        class_weights[dataset.classes[label]] for label in dataset.targets
    ]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    logger.info(
        "WeightedRandomSampler created – effective weights: %s",
        {k: f"{v:.2f}" for k, v in class_weights.items()},
    )
    return sampler


# ──────────────────────────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────────────────────────


def create_dataloaders(
    data_dir: Optional[Path] = None,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    use_weighted_sampler: bool = True,
) -> Dict[str, Tuple[DataLoader, AntiSpoofDataset]]:
    """Create train / val / test ``DataLoader`` instances.

    Parameters
    ----------
    data_dir : Path
        Root of the split dataset (must contain train/, val/, test/ sub-dirs).
    batch_size : int
        Mini-batch size.
    num_workers : int
        Number of parallel data-loading workers.
    use_weighted_sampler : bool
        If ``True``, the *training* loader uses ``WeightedRandomSampler``
        to oversample the minority class.

    Returns
    -------
    dict  { split_name: (DataLoader, AntiSpoofDataset) }
    """
    data_dir = data_dir or config.OUTPUT_DATA_DIR

    loaders: Dict[str, Tuple[DataLoader, AntiSpoofDataset]] = {}

    for split_name in ("train", "val", "test"):
        split_path = data_dir / split_name / "celeba-spoof"
        if not split_path.exists():
            logger.warning("Split directory '%s' does not exist – skipping.", split_path)
            continue

        tfm = get_train_transforms() if split_name == "train" else get_eval_transforms()
        ds = AntiSpoofDataset(split_path, transform=tfm)
        ds.print_distribution(split_name)

        sampler = None
        shuffle = False
        if split_name == "train":
            if use_weighted_sampler:
                sampler = get_weighted_sampler(ds)
            else:
                shuffle = True

        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=(split_name == "train"),
        )
        loaders[split_name] = (loader, ds)
        logger.info(
            "[%s] DataLoader ready – %d batches (batch_size=%d).",
            split_name, len(loader), batch_size,
        )

    return loaders
