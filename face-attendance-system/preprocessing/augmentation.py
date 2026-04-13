"""
augmentation.py – Data augmentation transforms for face anti-spoofing.

Training augmentations are intentionally *mild* to preserve spoof artifacts
such as printed-paper texture, moiré patterns, and screen reflections.

Provides:
  - get_train_transforms()
  - get_eval_transforms()   (shared by val & test)
"""

from torchvision import transforms

from . import config


def get_train_transforms() -> transforms.Compose:
    """Return the augmentation pipeline for training images.

    Augmentations applied:
      • Random horizontal flip (p=0.5)
      • Slight random rotation (±10°)
      • Mild brightness / contrast jitter
      • Light Gaussian blur
      • Resize to IMAGE_SIZE × IMAGE_SIZE
      • Convert to tensor
      • ImageNet normalisation
    """
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=config.ROTATION_DEGREES),
        transforms.ColorJitter(
            brightness=config.BRIGHTNESS_JITTER,
            contrast=config.CONTRAST_JITTER,
        ),
        transforms.GaussianBlur(
            kernel_size=config.GAUSSIAN_BLUR_KERNEL,
            sigma=config.GAUSSIAN_BLUR_SIGMA,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])


def get_eval_transforms() -> transforms.Compose:
    """Return the deterministic pipeline for validation / test images.

    Only resizing, tensor conversion, and normalisation — no randomness.
    """
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])
