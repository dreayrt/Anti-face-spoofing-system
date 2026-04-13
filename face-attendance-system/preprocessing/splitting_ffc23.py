"""
splitting_ffc23.py – Chia dữ liệu Train / Val / Test theo video cho FF-C23.

Tất cả frame của 1 video sẽ nằm trong cùng 1 split → không bị data leakage.
"""

import logging
from pathlib import Path
from typing import Dict, List

from sklearn.model_selection import train_test_split

from . import config_ffc23 as config

logger = logging.getLogger(__name__)


def collect_video_ids(raw_dir: Path) -> Dict[str, List[str]]:
    """Thu thập video ID cho mỗi loại (original + 5 spoof methods)."""
    video_map: Dict[str, List[str]] = {}

    real_dir = raw_dir / config.REAL_DIR_NAME
    if real_dir.is_dir():
        videos = sorted(
            f.stem for f in real_dir.iterdir()
            if f.is_file() and f.suffix.lower() in config.VIDEO_EXTENSIONS
        )
        video_map[config.REAL_DIR_NAME] = videos
        logger.info("  %-18s : %d videos", config.REAL_DIR_NAME, len(videos))

    for method in config.SPOOF_METHODS:
        method_dir = raw_dir / method
        if method_dir.is_dir():
            videos = sorted(
                f.stem for f in method_dir.iterdir()
                if f.is_file() and f.suffix.lower() in config.VIDEO_EXTENSIONS
            )
            video_map[method] = videos
            logger.info("  %-18s : %d videos", method, len(videos))

    return video_map


def split_video_ids(
    video_ids: List[str],
    category_name: str = "",
) -> Dict[str, List[str]]:
    """Chia danh sách video ID thành train/val/test."""
    if not video_ids:
        return {"train": [], "val": [], "test": []}

    val_test_ratio = config.VAL_RATIO + config.TEST_RATIO
    train_ids, valtest_ids = train_test_split(
        video_ids,
        test_size=val_test_ratio,
        random_state=config.RANDOM_SEED,
    )

    relative_test = config.TEST_RATIO / val_test_ratio
    val_ids, test_ids = train_test_split(
        valtest_ids,
        test_size=relative_test,
        random_state=config.RANDOM_SEED,
    )

    splits = {"train": train_ids, "val": val_ids, "test": test_ids}

    for name, ids in splits.items():
        logger.info(
            "  [%s] %-5s : %d videos",
            category_name or "unknown", name, len(ids),
        )

    return splits


def run_splitting(
    raw_dir: Path | None = None,
) -> Dict[str, Dict[str, List[str]]]:
    """Chia toàn bộ video thành train/val/test.

    Returns  { category: { split_name: [video_id, ...] } }
    """
    raw_dir = raw_dir or config.RAW_DATA_DIR

    logger.info("=" * 60)
    logger.info("CHIA DỮ LIỆU THEO VIDEO (VIDEO-LEVEL SPLIT)")
    logger.info("=" * 60)
    logger.info("Raw dir: %s", raw_dir)
    logger.info("Tỉ lệ : %.0f%% train / %.0f%% val / %.0f%% test",
                config.TRAIN_RATIO * 100, config.VAL_RATIO * 100, config.TEST_RATIO * 100)

    logger.info("")
    logger.info("Thu thập video IDs:")
    video_map = collect_video_ids(raw_dir)

    if not video_map:
        logger.error("Không tìm thấy video nào trong '%s'!", raw_dir)
        return {}

    all_splits: Dict[str, Dict[str, List[str]]] = {}
    logger.info("")
    logger.info("Chia video IDs:")

    for category, video_ids in video_map.items():
        all_splits[category] = split_video_ids(video_ids, category)

    logger.info("")
    logger.info("─" * 50)
    logger.info("TỔNG KẾT CHIA DỮ LIỆU:")
    for split_name in ("train", "val", "test"):
        total = sum(
            len(all_splits[cat].get(split_name, []))
            for cat in all_splits
        )
        logger.info("  %-5s : %d videos tổng cộng", split_name, total)
    logger.info("─" * 50)

    return all_splits
