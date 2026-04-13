"""
splitting.py – Train / Validation / Test splitting.

Combines all raw images into a single pool and performs a stratified
random split into 70% train / 15% val / 15% test.

NOTE: Identity-based splitting is NOT possible with this dataset because
the filenames do not encode subject identity.  A stratified random split
is used instead.  This means the same person *could* appear in both
train and test, which may inflate evaluation metrics slightly.
"""

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.model_selection import train_test_split

from . import config

logger = logging.getLogger(__name__)


def collect_all_images(raw_dir: Path) -> List[Tuple[Path, str]]:
    """Walk through *raw_dir* (which may have train/val sub-splits)
    and return a flat list of (image_path, class_name) tuples."""
    samples: List[Tuple[Path, str]] = []
    for class_name in config.CLASS_NAMES:
        for split_dir in raw_dir.iterdir():
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                continue
            for img_path in sorted(class_dir.iterdir()):
                if img_path.is_file() and img_path.suffix.lower() in config.IMAGE_EXTENSIONS:
                    samples.append((img_path, class_name))

    logger.info("Collected %d total images from '%s'.", len(samples), raw_dir)
    for cn in config.CLASS_NAMES:
        count = sum(1 for _, c in samples if c == cn)
        logger.info("  %-8s : %d", cn, count)
    return samples


def stratified_split(
    samples: List[Tuple[Path, str]],
) -> Dict[str, List[Tuple[Path, str]]]:
    """Perform a stratified random split.

    Returns dict with keys 'train', 'val', 'test',
    each mapping to a list of (path, class_name).
    """
    paths, labels = zip(*samples) if samples else ([], [])
    paths = list(paths)
    labels = list(labels)

    # First split: train vs (val + test)
    val_test_ratio = config.VAL_RATIO + config.TEST_RATIO
    train_paths, valtest_paths, train_labels, valtest_labels = train_test_split(
        paths, labels,
        test_size=val_test_ratio,
        stratify=labels,
        random_state=config.RANDOM_SEED,
    )

    # Second split: val vs test (split the remaining half-half)
    relative_test = config.TEST_RATIO / val_test_ratio
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        valtest_paths, valtest_labels,
        test_size=relative_test,
        stratify=valtest_labels,
        random_state=config.RANDOM_SEED,
    )

    splits = {
        "train": list(zip(train_paths, train_labels)),
        "val": list(zip(val_paths, val_labels)),
        "test": list(zip(test_paths, test_labels)),
    }

    for name, data in splits.items():
        logger.info("Split %-5s : %d images", name, len(data))
        for cn in config.CLASS_NAMES:
            count = sum(1 for _, c in data if c == cn)
            logger.info("  %-8s : %d", cn, count)

    return splits


def copy_to_output(
    splits: Dict[str, List[Tuple[Path, str]]],
    output_dir: Path | None = None,
) -> Path:
    """Copy split images into the canonical folder structure:
        output_dir / {train,val,test} / celeba-spoof / {live,spoof} /
    Returns the output_dir.
    """
    output_dir = output_dir or config.OUTPUT_DATA_DIR
    logger.info("Copying images to '%s' ...", output_dir)

    for split_name, data in splits.items():
        for class_name in config.CLASS_NAMES:
            dest = output_dir / split_name / "celeba-spoof" / class_name
            dest.mkdir(parents=True, exist_ok=True)

        for src_path, class_name in data:
            dest_path = output_dir / split_name / "celeba-spoof" / class_name / src_path.name
            # Handle duplicate filenames across raw sub-folders
            if dest_path.exists():
                stem = src_path.stem
                suffix = src_path.suffix
                counter = 1
                while dest_path.exists():
                    dest_path = output_dir / split_name / "celeba-spoof" / class_name / f"{stem}_dup{counter}{suffix}"
                    counter += 1
            shutil.copy2(src_path, dest_path)

    logger.info("Copy complete.")
    return output_dir


def run_splitting(raw_dir: Path | None = None, output_dir: Path | None = None) -> Dict[str, List[Tuple[Path, str]]]:
    """Full splitting pipeline: collect → split → copy."""
    raw_dir = raw_dir or config.RAW_DATA_DIR
    output_dir = output_dir or config.OUTPUT_DATA_DIR

    logger.info("=" * 60)
    logger.info("STARTING DATA SPLITTING")
    logger.info("=" * 60)
    logger.warning(
        "Identity-based splitting is NOT available (no identity labels in filenames). "
        "Using stratified RANDOM split instead. The same person may appear in "
        "multiple splits, which could cause slight data leakage."
    )

    samples = collect_all_images(raw_dir)
    if not samples:
        logger.error("No images found in '%s'. Aborting split.", raw_dir)
        return {}

    splits = stratified_split(samples)
    copy_to_output(splits, output_dir)
    return splits
