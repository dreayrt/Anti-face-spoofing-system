"""
augmentation_siw.py – Data augmentation transforms cho SiW dataset.

Augmentation nhẹ, bảo toàn spoof artifacts (moiré, screen edge, paper texture).

Provides:
  - get_train_transforms()
  - get_eval_transforms()   (shared by val & test)
"""

import torch
from torchvision import transforms

from . import config_siw as config


# ──────────────────────────────────────────────────────────────
# Custom Transform: Gaussian Noise nhẹ
# ──────────────────────────────────────────────────────────────


class AddGaussianNoise:
    """Thêm Gaussian noise nhẹ vào tensor ảnh.

    Parameters
    ----------
    mean : float
        Trung bình của noise (thường = 0).
    std : float
        Độ lệch chuẩn của noise (nhỏ = nhẹ).
    """

    def __init__(self, mean: float = 0.0, std: float = 0.01):
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


# ──────────────────────────────────────────────────────────────
# Train Transforms
# ──────────────────────────────────────────────────────────────


def get_train_transforms() -> transforms.Compose:
    """Return augmentation pipeline cho training images SiW.

    Augmentation nhẹ theo yêu cầu:
      • Random horizontal flip (p=0.5)
      • Brightness nhẹ (±0.15)
      • Contrast nhẹ (±0.15)
      • Gaussian noise nhẹ (σ=0.01)
      • Resize to IMAGE_SIZE × IMAGE_SIZE
      • Convert to tensor
      • ImageNet normalisation
    """
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=config.BRIGHTNESS_JITTER,
            contrast=config.CONTRAST_JITTER,
        ),
        transforms.ToTensor(),
        AddGaussianNoise(mean=0.0, std=config.GAUSSIAN_NOISE_STD),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])


# ──────────────────────────────────────────────────────────────
# Eval Transforms (Val / Test)
# ──────────────────────────────────────────────────────────────


def get_eval_transforms() -> transforms.Compose:
    """Return deterministic pipeline cho validation / test images.

    Chỉ resize, tensor conversion, và normalisation — không augmentation.
    """
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])
