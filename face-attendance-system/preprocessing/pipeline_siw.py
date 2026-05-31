"""
pipeline_siw.py – Điều phối pipeline tiền xử lý SiW.

Cách chạy (từ project root):
    python -m preprocessing.pipeline_siw

Các bước:
  1. Face Alignment (MTCNN verify + align + resize)
  2. Data Cleaning (xoá ảnh corrupt, mờ)
  3. Copy vào output (giữ nguyên split từ raw)
  4. Thống kê + tạo DataLoaders
  5. Visualize augmented samples
  6. Class distribution chart
"""

import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Dict

from . import config_siw as config
from .face_alignment import process_directory
from .dataset_siw import create_dataloaders, get_class_weights

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────────────────────


def _setup_logging() -> None:
    """Configure root logger -> console + log file."""
    log_path = config.OUTPUTS_DIR / "pipeline_siw.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    fmt = "[%(asctime)s] %(levelname)-8s %(name)s - %(message)s"

    # Use UTF-8 for console on Windows to avoid cp1252 encoding errors
    import io
    utf8_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    console_handler = logging.StreamHandler(utf8_stdout)
    file_handler = logging.FileHandler(str(log_path), mode="w", encoding="utf-8")

    handlers = [console_handler, file_handler]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)


# ──────────────────────────────────────────────────────────────
# Cleaning nhẹ (corrupt + blur)
# ──────────────────────────────────────────────────────────────


def _run_cleaning_on_dir(directory: Path) -> Dict[str, int]:
    """Xoá ảnh corrupt + mờ trong thư mục."""
    from PIL import Image
    import cv2

    image_files = sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS
    )

    corrupted = 0
    blurry = 0

    for path in image_files:
        # Check corrupt
        try:
            with Image.open(path) as img:
                img.verify()
            with Image.open(path) as img:
                img.load()
        except Exception:
            path.unlink(missing_ok=True)
            corrupted += 1
            continue

        # Check blur
        img_gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img_gray is not None:
            blur_score = cv2.Laplacian(img_gray, cv2.CV_64F).var()
            if blur_score < config.BLUR_THRESHOLD:
                path.unlink(missing_ok=True)
                blurry += 1

    return {"corrupted": corrupted, "blurry": blurry}


# ──────────────────────────────────────────────────────────────
# Copy processed images to output
# ──────────────────────────────────────────────────────────────


def _copy_to_output(
    processed_dir: Path,
    output_dir: Path,
    split_name: str,
) -> Dict[str, int]:
    """Copy ảnh đã xử lý vào output structure.

    processed_dir chứa ảnh đã align/resize.
    Output: output_dir / split_name / SiW / {live, spoof} /
    """
    counts = {"live": 0, "spoof": 0}

    for raw_class, output_class in config.RAW_CLASS_MAP.items():
        src_dir = processed_dir / split_name / raw_class
        if not src_dir.is_dir():
            logger.warning("  Not found: %s", src_dir)
            continue

        dest_dir = output_dir / split_name / config.DATASET_NAME / output_class
        dest_dir.mkdir(parents=True, exist_ok=True)

        image_files = sorted(
            p for p in src_dir.iterdir()
            if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS
        )

        for src_path in image_files:
            dest_path = dest_dir / src_path.name
            # Handle duplicate filenames
            if dest_path.exists():
                stem = src_path.stem
                suffix = src_path.suffix
                counter = 1
                while dest_path.exists():
                    dest_path = dest_dir / f"{stem}_dup{counter}{suffix}"
                    counter += 1
            shutil.copy2(src_path, dest_path)
            counts[output_class] += 1

    return counts


# ──────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────


def run_pipeline() -> None:
    _setup_logging()
    t0 = time.time()

    logger.info("=" * 70)
    logger.info("  SiW PREPROCESSING PIPELINE")
    logger.info("=" * 70)
    logger.info("Raw data dir  : %s", config.RAW_DATA_DIR)
    logger.info("Output dir    : %s", config.OUTPUT_DATA_DIR)
    logger.info("Dataset name  : %s", config.DATASET_NAME)
    logger.info("Image size    : %dx%d", config.IMAGE_SIZE, config.IMAGE_SIZE)
    logger.info("Strategy      : Keep original split (train/val/test) from raw data")

    # Kiểm tra raw data tồn tại
    if not config.RAW_DATA_DIR.exists():
        logger.error("Raw data directory not found: %s", config.RAW_DATA_DIR)
        return

    # Tạo thư mục tạm cho ảnh đã xử lý
    temp_dir = config.OUTPUTS_DIR / "siw_processed"

    # ── Bước 1: Face Alignment ──────────────────────────────
    logger.info("")
    logger.info(">> STEP 1/6 - Face Alignment (MTCNN Verify + Align + Resize)")

    alignment_stats = {}
    for split_name in ("train", "val", "test"):
        for raw_class in config.RAW_CLASS_MAP.keys():
            input_dir = config.RAW_DATA_DIR / split_name / raw_class
            if not input_dir.is_dir():
                logger.warning("  Skip: %s (not found)", input_dir)
                continue

            output_class = config.RAW_CLASS_MAP[raw_class]
            out_dir = temp_dir / split_name / raw_class

            logger.info("")
            logger.info("  [%s/%s] (%s -> %s)", split_name, raw_class, raw_class, output_class)
            stats = process_directory(
                input_dir, out_dir,
                output_size=config.ALIGNMENT_OUTPUT_SIZE,
                margin=config.FACE_MARGIN,
            )
            alignment_stats[f"{split_name}/{raw_class}"] = stats

    # ── Bước 2: Data Cleaning ────────────────────────────────
    logger.info("")
    logger.info(">> STEP 2/6 - Data Cleaning (Corrupt + Blur)")

    total_corrupted = 0
    total_blurry = 0
    for split_name in ("train", "val", "test"):
        split_dir = temp_dir / split_name
        if not split_dir.exists():
            continue
        logger.info("  Cleaning [%s]...", split_name)
        cleaning = _run_cleaning_on_dir(split_dir)
        total_corrupted += cleaning["corrupted"]
        total_blurry += cleaning["blurry"]
        if cleaning["corrupted"] > 0 or cleaning["blurry"] > 0:
            logger.info("    Removed: %d corrupt, %d blur", cleaning["corrupted"], cleaning["blurry"])

    logger.info("  Total cleaning: %d corrupt + %d blur removed", total_corrupted, total_blurry)

    # ── Bước 3: Copy vào output (giữ nguyên split) ──────────
    logger.info("")
    logger.info(">> STEP 3/6 - Copy to Output (Keep original split)")

    total_counts = {}
    for split_name in ("train", "val", "test"):
        counts = _copy_to_output(temp_dir, config.OUTPUT_DATA_DIR, split_name)
        total_counts[split_name] = counts
        logger.info(
            "  [%s] Copied: %d live, %d spoof",
            split_name, counts["live"], counts["spoof"],
        )

    # ── Bước 4: Thống kê ─────────────────────────────────────
    logger.info("")
    logger.info(">> STEP 4/6 - Summary Statistics")
    _print_summary(alignment_stats, total_counts, total_corrupted, total_blurry, logger)

    # ── Bước 5: DataLoaders ──────────────────────────────────
    logger.info("")
    logger.info(">> STEP 5/6 - Create PyTorch DataLoaders")
    try:
        loaders = create_dataloaders()
        if "train" in loaders:
            _, train_ds = loaders["train"]
            weights = get_class_weights(train_ds)
            logger.info("Suggested loss weights (CrossEntropyLoss): %s", weights.tolist())
    except Exception as exc:
        logger.warning("Failed to create DataLoaders: %s", exc)
        loaders = {}

    # ── Bước 6: Visualize ────────────────────────────────────
    logger.info("")
    logger.info(">> STEP 6/6 - Visualize")
    try:
        from .visualization import visualize_augmented_samples, plot_class_distribution

        for split_name in ("train", "val", "test"):
            if split_name in loaders:
                _, ds = loaders[split_name]
                save_path = config.OUTPUTS_DIR / f"augmented_samples_siw_{split_name}.png"
                visualize_augmented_samples(ds, n_per_class=8, split_name=f"SiW-{split_name}", save_path=save_path)

        split_counts = {}
        for split_name in ("train", "val", "test"):
            if split_name in loaders:
                _, ds = loaders[split_name]
                split_counts[split_name] = ds.class_counts()
        if split_counts:
            save_path = config.OUTPUTS_DIR / "class_distribution_siw.png"
            plot_class_distribution(split_counts, save_path=save_path)
    except Exception as exc:
        logger.warning("Failed to visualize: %s", exc)

    # ── Dọn dẹp thư mục tạm ─────────────────────────────────
    logger.info("")
    logger.info("Cleaning up temp directory...")
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.info("Deleted temp directory: %s", temp_dir)
    except Exception:
        logger.warning("Failed to delete temp directory: %s", temp_dir)

    elapsed = time.time() - t0
    logger.info("")
    logger.info("=" * 70)
    logger.info("  SiW PIPELINE COMPLETED (%.1f s)", elapsed)
    logger.info("=" * 70)
    logger.info("Output dataset : %s", config.OUTPUT_DATA_DIR)
    logger.info("Output folders :")
    for split_name in ("train", "val", "test"):
        siw_dir = config.OUTPUT_DATA_DIR / split_name / config.DATASET_NAME
        logger.info("  %s", siw_dir)
    logger.info("Log file       : %s", config.OUTPUTS_DIR / "pipeline_siw.log")


def _print_summary(alignment_stats, total_counts, total_corrupted, total_blurry, logger):
    """In bảng tổng kết."""
    logger.info("-" * 50)

    logger.info("FACE ALIGNMENT:")
    for key, stats in alignment_stats.items():
        logger.info(
            "  %-20s : %d total, %d processed, %d failed",
            key, stats["total"], stats["aligned"] + stats["resized_only"], stats["failed"],
        )

    logger.info("")
    logger.info("CLEANING:")
    logger.info("  Corrupted removed : %d", total_corrupted)
    logger.info("  Blurry removed    : %d", total_blurry)

    logger.info("")
    logger.info("DATASET OUTPUT:")
    grand_live = 0
    grand_spoof = 0
    for split_name in ("train", "val", "test"):
        counts = total_counts.get(split_name, {"live": 0, "spoof": 0})
        live = counts["live"]
        spoof = counts["spoof"]
        total = live + spoof
        grand_live += live
        grand_spoof += spoof
        logger.info(
            "  %-5s : %5d images (live=%d, spoof=%d)",
            split_name, total, live, spoof,
        )
    logger.info(
        "  TOTAL : %5d images (live=%d, spoof=%d)",
        grand_live + grand_spoof, grand_live, grand_spoof,
    )
    logger.info("-" * 50)


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_pipeline()
