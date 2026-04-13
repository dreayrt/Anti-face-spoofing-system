"""
cleaning.py – Data cleaning utilities.

Handles:
  1. Corrupted / unreadable image removal
  2. Duplicate image detection via perceptual hashing (dhash)
  3. Face detection using OpenCV Haar Cascade (optional removal)
"""

import logging
from pathlib import Path
from typing import List, Tuple, Dict, Set

from PIL import Image
import imagehash
import cv2
import numpy as np

from . import config

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# 1. Corrupted image detection
# ──────────────────────────────────────────────────────────────


def find_corrupted_images(directory: Path) -> List[Path]:
    """Scan *directory* recursively and return a list of corrupted / unreadable image paths."""
    corrupted: List[Path] = []
    image_files = _collect_image_paths(directory)
    logger.info("Scanning %d images for corruption in '%s' ...", len(image_files), directory)

    for path in image_files:
        try:
            with Image.open(path) as img:
                img.verify()
            # Re-open to make sure data is actually decodable
            with Image.open(path) as img:
                img.load()
        except Exception as exc:
            logger.warning("Corrupted image: %s (%s)", path, exc)
            corrupted.append(path)

    logger.info("Found %d corrupted images.", len(corrupted))
    return corrupted


def remove_corrupted_images(directory: Path) -> int:
    """Detect and delete corrupted images.  Returns count of removed files."""
    corrupted = find_corrupted_images(directory)
    for path in corrupted:
        path.unlink(missing_ok=True)
        logger.info("Removed corrupted: %s", path)
    return len(corrupted)


# ──────────────────────────────────────────────────────────────
# 2. Duplicate detection (perceptual hash)
# ──────────────────────────────────────────────────────────────


def compute_hashes(directory: Path) -> Dict[str, List[Path]]:
    """Compute dhash for every image and group by hash string.

    Returns a dict  { hash_hex: [path1, path2, ...] }.
    Groups with len > 1 are duplicates.
    """
    hash_map: Dict[str, List[Path]] = {}
    image_files = _collect_image_paths(directory)
    logger.info("Computing perceptual hashes for %d images ...", len(image_files))

    for path in image_files:
        try:
            with Image.open(path) as img:
                h = str(imagehash.dhash(img, hash_size=config.HASH_SIZE))
        except Exception:
            continue
        hash_map.setdefault(h, []).append(path)
    return hash_map


def find_duplicates(directory: Path) -> List[Tuple[Path, List[Path]]]:
    """Return list of (kept_path, [duplicate_paths]) tuples."""
    hash_map = compute_hashes(directory)
    duplicates: List[Tuple[Path, List[Path]]] = []
    for h, paths in hash_map.items():
        if len(paths) > 1:
            # Keep the first, flag the rest
            duplicates.append((paths[0], paths[1:]))
    total_dup = sum(len(dups) for _, dups in duplicates)
    logger.info("Found %d duplicate groups (%d extra images).", len(duplicates), total_dup)
    return duplicates


def remove_duplicates(directory: Path) -> int:
    """Detect and delete duplicate images. Returns count of removed files."""
    duplicates = find_duplicates(directory)
    removed = 0
    for kept, dups in duplicates:
        for dup in dups:
            dup.unlink(missing_ok=True)
            logger.info("Removed duplicate: %s  (kept %s)", dup, kept)
            removed += 1
    return removed


# ──────────────────────────────────────────────────────────────
# 3. Face detection (Haar Cascade)
# ──────────────────────────────────────────────────────────────

_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore[attr-defined]


def find_images_without_face(directory: Path) -> List[Path]:
    """Return images where Haar Cascade detects no face."""
    cascade = cv2.CascadeClassifier(_CASCADE_PATH)
    if cascade.empty():
        logger.error("Failed to load Haar Cascade from %s", _CASCADE_PATH)
        return []

    image_files = _collect_image_paths(directory)
    no_face: List[Path] = []
    logger.info("Running face detection on %d images ...", len(image_files))

    for path in image_files:
        try:
            img = cv2.imread(str(path))
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            if len(faces) == 0:
                no_face.append(path)
        except Exception as exc:
            logger.warning("Face detection error on %s: %s", path, exc)

    logger.info("Found %d images with no detectable face.", len(no_face))
    return no_face


def remove_no_face_images(directory: Path) -> int:
    """Optionally remove images without a detectable face."""
    if not config.REMOVE_NO_FACE:
        no_face = find_images_without_face(directory)
        logger.info(
            "REMOVE_NO_FACE is False — %d images without faces will be KEPT. "
            "Set config.REMOVE_NO_FACE = True to delete them.",
            len(no_face),
        )
        return 0

    no_face = find_images_without_face(directory)
    for path in no_face:
        path.unlink(missing_ok=True)
        logger.info("Removed (no face): %s", path)
    return len(no_face)


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────


def _collect_image_paths(directory: Path) -> List[Path]:
    """Recursively collect all image files matching allowed extensions."""
    return sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS
    )


def run_full_cleaning(directory: Path) -> dict:
    """Run the complete cleaning pipeline and return a summary dict."""
    logger.info("=" * 60)
    logger.info("STARTING DATA CLEANING on '%s'", directory)
    logger.info("=" * 60)

    summary = {
        "corrupted_removed": remove_corrupted_images(directory),
        "duplicates_removed": remove_duplicates(directory),
        "no_face_removed": remove_no_face_images(directory),
    }

    logger.info("Cleaning summary: %s", summary)
    return summary
