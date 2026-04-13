"""
visualization.py – Visualization utilities for the preprocessing pipeline.

Provides:
  - visualize_augmented_samples()   grid of augmented images per class
  - plot_class_distribution()        bar chart of class counts per split
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid

from . import config
from .dataset import AntiSpoofDataset

logger = logging.getLogger(__name__)


def _denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalisation so the image looks natural."""
    mean = torch.tensor(config.IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(config.IMAGENET_STD).view(3, 1, 1)
    return torch.clamp(tensor * std + mean, 0.0, 1.0)


def visualize_augmented_samples(
    dataset: AntiSpoofDataset,
    n_per_class: int = 8,
    save_path: Optional[Path] = None,
    split_name: str = "train",
) -> Path:
    """Display and save a grid of augmented samples from each class.

    Parameters
    ----------
    dataset : AntiSpoofDataset
        Dataset with transforms already applied.
    n_per_class : int
        Number of samples to show per class.
    save_path : Path, optional
        Where to save the figure. Defaults to ``outputs/augmented_samples_{split}.png``.
    split_name : str
        Used in the figure title and default filename.

    Returns
    -------
    Path to the saved figure.
    """
    save_path = save_path or config.OUTPUTS_DIR / f"augmented_samples_{split_name}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    images_by_class: Dict[str, list] = {cls: [] for cls in dataset.classes}
    indices = torch.randperm(len(dataset)).tolist()

    for idx in indices:
        img, label = dataset[idx]
        cls_name = dataset.classes[label]
        if len(images_by_class[cls_name]) < n_per_class:
            images_by_class[cls_name].append(_denormalize(img))
        if all(len(v) >= n_per_class for v in images_by_class.values()):
            break

    fig, axes = plt.subplots(
        len(dataset.classes), 1,
        figsize=(n_per_class * 2.5, len(dataset.classes) * 2.8),
    )
    if len(dataset.classes) == 1:
        axes = [axes]

    for ax, cls_name in zip(axes, dataset.classes):
        imgs = images_by_class[cls_name]
        if imgs:
            grid = make_grid(imgs, nrow=n_per_class, padding=2)
            ax.imshow(grid.permute(1, 2, 0).numpy())
        ax.set_title(f"{cls_name} ({len(imgs)} samples)", fontsize=12, fontweight="bold")
        ax.axis("off")

    fig.suptitle(f"Augmented Samples – {split_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved augmented-sample visualization → %s", save_path)
    return save_path


def plot_class_distribution(
    split_counts: Dict[str, Dict[str, int]],
    save_path: Optional[Path] = None,
) -> Path:
    """Bar chart comparing live vs spoof counts across splits.

    Parameters
    ----------
    split_counts : dict
        ``{ split_name: { class_name: count } }``
    save_path : Path, optional
        Defaults to ``outputs/class_distribution.png``.

    Returns
    -------
    Path to the saved figure.
    """
    save_path = save_path or config.OUTPUTS_DIR / "class_distribution.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    splits = list(split_counts.keys())
    classes = config.CLASS_NAMES
    x = np.arange(len(splits))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2ecc71", "#e74c3c"]  # green for live, red for spoof

    for i, cls in enumerate(classes):
        values = [split_counts[s].get(cls, 0) for s in splits]
        bars = ax.bar(x + i * width, values, width, label=cls, color=colors[i], edgecolor="white")
        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                str(v), ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

    ax.set_xlabel("Split", fontsize=12)
    ax.set_ylabel("Number of Images", fontsize=12)
    ax.set_title("Class Distribution per Split", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(splits, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved class-distribution chart → %s", save_path)
    return save_path
