"""
pipeline.py – Main orchestrator for the face anti-spoofing preprocessing pipeline.

Usage (from project root):
    python -m preprocessing.pipeline

Steps executed:
  1. Clean raw data  (corrupted files, duplicates, optional face detection)
  2. Split into train / val / test  (70 / 15 / 15, stratified random)
  3. Print dataset summary statistics
  4. Create PyTorch DataLoaders  (with WeightedRandomSampler)
  5. Visualize augmented samples
  6. Save class-distribution chart
"""

import logging
import sys
import time
from pathlib import Path

from . import config
from .cleaning import run_full_cleaning
from .splitting import run_splitting
from .dataset import create_dataloaders, get_class_weights
from .visualization import plot_class_distribution, visualize_augmented_samples

# ──────────────────────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────────────────────


def _setup_logging() -> None:
    """Configure root logger to write to console and a log file."""
    log_path = config.OUTPUTS_DIR / "pipeline.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    fmt = "[%(asctime)s] %(levelname)-8s %(name)s – %(message)s"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(log_path), mode="w", encoding="utf-8"),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)


# ──────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────


def run_pipeline() -> None:
    _setup_logging()
    logger = logging.getLogger(__name__)
    t0 = time.time()

    logger.info("=" * 70)
    logger.info("  FACE ANTI-SPOOFING PREPROCESSING PIPELINE")
    logger.info("=" * 70)
    logger.info("Raw data dir  : %s", config.RAW_DATA_DIR)
    logger.info("Output dir    : %s", config.OUTPUT_DATA_DIR)
    logger.info("Image size    : %d×%d", config.IMAGE_SIZE, config.IMAGE_SIZE)
    logger.info("Split ratios  : %.0f%% / %.0f%% / %.0f%%",
                config.TRAIN_RATIO * 100, config.VAL_RATIO * 100, config.TEST_RATIO * 100)

    # ── Step 1: Clean ─────────────────────────────────────────
    logger.info("")
    logger.info("▶  STEP 1 / 6 – Data Cleaning")
    cleaning_summary = run_full_cleaning(config.RAW_DATA_DIR)

    # ── Step 2: Split ─────────────────────────────────────────
    logger.info("")
    logger.info("▶  STEP 2 / 6 – Data Splitting")
    splits = run_splitting()

    # ── Step 3: Summary statistics ────────────────────────────
    logger.info("")
    logger.info("▶  STEP 3 / 6 – Summary Statistics")
    _print_summary(splits, cleaning_summary, logger)

    # ── Step 4: DataLoaders ───────────────────────────────────
    logger.info("")
    logger.info("▶  STEP 4 / 6 – Creating PyTorch DataLoaders")
    loaders = create_dataloaders()

    # Print class weights for the loss function
    if "train" in loaders:
        _, train_ds = loaders["train"]
        weights = get_class_weights(train_ds)
        logger.info("Recommended loss weights (CrossEntropyLoss): %s", weights.tolist())

    # ── Step 5: Visualize augmented samples ───────────────────
    logger.info("")
    logger.info("▶  STEP 5 / 6 – Visualizing Augmented Samples")
    for split_name in ("train", "val", "test"):
        if split_name in loaders:
            _, ds = loaders[split_name]
            visualize_augmented_samples(ds, n_per_class=8, split_name=split_name)

    # ── Step 6: Class distribution chart ──────────────────────
    logger.info("")
    logger.info("▶  STEP 6 / 6 – Plotting Class Distribution")
    split_counts = {}
    for split_name in ("train", "val", "test"):
        if split_name in loaders:
            _, ds = loaders[split_name]
            split_counts[split_name] = ds.class_counts()
    if split_counts:
        plot_class_distribution(split_counts)

    elapsed = time.time() - t0
    logger.info("")
    logger.info("=" * 70)
    logger.info("  PIPELINE COMPLETE  (%.1f s)", elapsed)
    logger.info("=" * 70)
    logger.info("Output dataset : %s", config.OUTPUT_DATA_DIR)
    logger.info("Visualizations : %s", config.OUTPUTS_DIR)


def _print_summary(splits, cleaning_summary, logger):
    """Print a consolidated summary table."""
    logger.info("─" * 50)
    logger.info("Cleaning results:")
    for key, val in cleaning_summary.items():
        logger.info("  %-22s : %d", key, val)

    logger.info("")
    logger.info("Split sizes:")
    total = 0
    for split_name in ("train", "val", "test"):
        data = splits.get(split_name, [])
        n = len(data)
        total += n
        live_count = sum(1 for _, c in data if c == "live")
        spoof_count = sum(1 for _, c in data if c == "spoof")
        logger.info(
            "  %-5s : %5d  (live=%d, spoof=%d)",
            split_name, n, live_count, spoof_count,
        )
    logger.info("  TOTAL : %5d", total)
    logger.info("─" * 50)


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_pipeline()
