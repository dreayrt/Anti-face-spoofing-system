"""
cleaning_ffc23.py – Data cleaning nhẹ cho frame FF-C23 đã trích xuất.

Xử lý:
  1. Xoá ảnh bị corrupt / không đọc được
  2. Xoá ảnh bị mờ nặng (Laplacian variance < threshold)
"""

import logging
from pathlib import Path
from typing import Dict, List

import cv2
from PIL import Image

from . import config_ffc23 as config

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# 1. Phát hiện ảnh bị corrupt
# ──────────────────────────────────────────────────────────────


def find_corrupted_images(directory: Path) -> List[Path]:
    """Quét thư mục và trả về danh sách ảnh bị corrupt."""
    corrupted: List[Path] = []
    image_files = _collect_image_paths(directory)
    logger.info("  Quét %d ảnh tìm file corrupt...", len(image_files))

    for path in image_files:
        try:
            with Image.open(path) as img:
                img.verify()
            with Image.open(path) as img:
                img.load()
        except Exception as exc:
            logger.debug("  Ảnh corrupt: %s (%s)", path.name, exc)
            corrupted.append(path)

    if corrupted:
        logger.info("  Tìm thấy %d ảnh corrupt.", len(corrupted))
    return corrupted


def remove_corrupted_images(directory: Path) -> int:
    """Phát hiện và xoá ảnh corrupt. Trả về số file đã xoá."""
    corrupted = find_corrupted_images(directory)
    for path in corrupted:
        path.unlink(missing_ok=True)
    if corrupted:
        logger.info("  Đã xoá %d ảnh corrupt.", len(corrupted))
    return len(corrupted)


# ──────────────────────────────────────────────────────────────
# 2. Phát hiện ảnh bị mờ
# ──────────────────────────────────────────────────────────────


def compute_blur_score(image_path: Path) -> float:
    """Tính Laplacian variance (càng thấp → ảnh càng mờ)."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return cv2.Laplacian(img, cv2.CV_64F).var()


def find_blurry_images(
    directory: Path,
    threshold: float = config.BLUR_THRESHOLD,
) -> List[Path]:
    """Trả về danh sách ảnh bị mờ."""
    blurry: List[Path] = []
    image_files = _collect_image_paths(directory)
    logger.info("  Quét %d ảnh tìm ảnh mờ (threshold=%.1f)...", len(image_files), threshold)

    for path in image_files:
        score = compute_blur_score(path)
        if score < threshold:
            blurry.append(path)

    if blurry:
        logger.info("  Tìm thấy %d ảnh mờ.", len(blurry))
    return blurry


def remove_blurry_images(
    directory: Path,
    threshold: float = config.BLUR_THRESHOLD,
) -> int:
    """Phát hiện và xoá ảnh mờ. Trả về số file đã xoá."""
    blurry = find_blurry_images(directory, threshold)
    for path in blurry:
        path.unlink(missing_ok=True)
    if blurry:
        logger.info("  Đã xoá %d ảnh mờ.", len(blurry))
    return len(blurry)


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────


def _collect_image_paths(directory: Path) -> List[Path]:
    """Thu thập đệ quy tất cả file ảnh."""
    return sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS
    )


# ──────────────────────────────────────────────────────────────
# Pipeline cleaning
# ──────────────────────────────────────────────────────────────


def run_cleaning(output_dir: Path | None = None) -> Dict[str, int]:
    """Chạy toàn bộ cleaning trên dataset FF-C23 đã trích xuất."""
    output_dir = output_dir or config.OUTPUT_DATA_DIR

    logger.info("=" * 60)
    logger.info("DATA CLEANING FF-C23 (NHẸ)")
    logger.info("=" * 60)

    total_corrupted = 0
    total_blurry = 0

    for split_name in ("train", "val", "test"):
        split_dir = output_dir / split_name / config.DATASET_NAME
        if not split_dir.exists():
            continue
        logger.info("")
        logger.info("▶ Cleaning [%s]...", split_name)
        total_corrupted += remove_corrupted_images(split_dir)
        total_blurry += remove_blurry_images(split_dir)

    summary = {
        "corrupted_removed": total_corrupted,
        "blurry_removed": total_blurry,
    }
    logger.info("")
    logger.info("Tổng kết cleaning: %s", summary)
    return summary
