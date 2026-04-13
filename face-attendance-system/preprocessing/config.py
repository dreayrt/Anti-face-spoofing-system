"""
config.py – Central configuration for the preprocessing pipeline.
All paths, hyperparameters, and constants are defined here.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "anti-spoof" / "raw" / "celeba-spoof"
OUTPUT_DATA_DIR = PROJECT_ROOT / "dataset"
OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"

# Ensure output directories exist
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Image settings
# ──────────────────────────────────────────────
IMAGE_SIZE = 224
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ──────────────────────────────────────────────
# Data splitting
# ──────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# ──────────────────────────────────────────────
# Augmentation parameters
# ──────────────────────────────────────────────
ROTATION_DEGREES = 10
BRIGHTNESS_JITTER = 0.2
CONTRAST_JITTER = 0.2
GAUSSIAN_BLUR_KERNEL = 3
GAUSSIAN_BLUR_SIGMA = (0.1, 1.0)

# ──────────────────────────────────────────────
# DataLoader
# ──────────────────────────────────────────────
BATCH_SIZE = 32
NUM_WORKERS = min(4, os.cpu_count() or 1)

# ──────────────────────────────────────────────
# Class labels
# ──────────────────────────────────────────────
CLASS_NAMES = ["live", "spoof"]
LIVE_LABEL = 0
SPOOF_LABEL = 1

# ──────────────────────────────────────────────
# Cleaning
# ──────────────────────────────────────────────
HASH_SIZE = 8           # dhash size for duplicate detection
HASH_THRESHOLD = 5      # Hamming-distance threshold for near-duplicates
REMOVE_NO_FACE = False  # If True, remove images with no detected face
