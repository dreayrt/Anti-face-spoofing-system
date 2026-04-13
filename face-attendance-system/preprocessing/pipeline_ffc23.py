"""
pipeline_ffc23.py – Điều phối pipeline tiền xử lý FF-C23.

Cách chạy (từ project root):
    python -m preprocessing.pipeline_ffc23

Các bước:
  1. Chia video thành train / val / test (theo video ID)
  2. Trích xuất frame + cắt mặt (MTCNN)
  3. Cleaning nhẹ (xoá ảnh corrupt, mờ)
  4. In thống kê + tạo DataLoaders
"""

import logging
import sys
import time
from pathlib import Path

from . import config_ffc23 as config
from .splitting_ffc23 import run_splitting
from .frame_extraction import run_extraction
from .cleaning_ffc23 import run_cleaning
from .dataset_ffc23 import create_dataloaders, get_class_weights

# ──────────────────────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────────────────────


def _setup_logging() -> None:
    """Cấu hình root logger → console + log file."""
    log_path = config.OUTPUTS_DIR / "pipeline_ffc23.log"
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
    logger.info("  FF-C23 PREPROCESSING PIPELINE")
    logger.info("=" * 70)
    logger.info("Raw data dir  : %s", config.RAW_DATA_DIR)
    logger.info("Output dir    : %s", config.OUTPUT_DATA_DIR)
    logger.info("Dataset name  : %s", config.DATASET_NAME)
    logger.info("Image size    : %d×%d", config.IMAGE_SIZE, config.IMAGE_SIZE)
    logger.info("Frame rate    : mỗi %d frame lấy 1", config.FRAME_SAMPLE_RATE)
    logger.info("Max frames    : %d frame/video", config.MAX_FRAMES_PER_VIDEO)
    logger.info("Split ratios  : %.0f%% / %.0f%% / %.0f%%",
                config.TRAIN_RATIO * 100, config.VAL_RATIO * 100, config.TEST_RATIO * 100)

    # ── Bước 1: Chia video ────────────────────────────────────
    logger.info("")
    logger.info("▶  BƯỚC 1 / 4 – Chia Dữ Liệu (Video-Level Split)")
    all_splits = run_splitting()

    if not all_splits:
        logger.error("Không có video nào để xử lý. Dừng pipeline.")
        return

    # ── Bước 2: Trích xuất frame + crop mặt ──────────────────
    logger.info("")
    logger.info("▶  BƯỚC 2 / 4 – Trích Xuất Frame + Cắt Mặt")
    extraction_stats = run_extraction(all_splits)

    # ── Bước 3: Cleaning ─────────────────────────────────────
    logger.info("")
    logger.info("▶  BƯỚC 3 / 4 – Data Cleaning (Nhẹ)")
    cleaning_summary = run_cleaning()

    # ── Bước 4: Thống kê & DataLoaders ───────────────────────
    logger.info("")
    logger.info("▶  BƯỚC 4 / 4 – Thống Kê & DataLoaders")
    _print_summary(extraction_stats, cleaning_summary, logger)

    try:
        loaders = create_dataloaders()
        if "train" in loaders:
            _, train_ds = loaders["train"]
            weights = get_class_weights(train_ds)
            logger.info("Loss weights đề xuất (CrossEntropyLoss): %s", weights.tolist())
    except Exception as exc:
        logger.warning("Không tạo được DataLoaders: %s", exc)

    elapsed = time.time() - t0
    logger.info("")
    logger.info("=" * 70)
    logger.info("  PIPELINE HOÀN TẤT  (%.1f s)", elapsed)
    logger.info("=" * 70)
    logger.info("Output dataset : %s", config.OUTPUT_DATA_DIR)
    logger.info("Log file       : %s", config.OUTPUTS_DIR / "pipeline_ffc23.log")


def _print_summary(extraction_stats, cleaning_summary, logger):
    """In bảng tổng kết."""
    logger.info("─" * 50)
    logger.info("THỐNG KÊ TRÍCH XUẤT:")
    for split_name in ("train", "val", "test"):
        stats = extraction_stats.get(split_name, {})
        total = sum(stats.values())
        live = stats.get("live", 0)
        spoof = sum(v for k, v in stats.items() if k != "live")
        logger.info("  %-5s : %5d frames (live=%d, spoof=%d)", split_name, total, live, spoof)
        for method in config.SPOOF_METHODS:
            if method in stats:
                logger.info("    %-18s : %d", method, stats[method])

    logger.info("")
    logger.info("KẾT QUẢ CLEANING:")
    for key, val in cleaning_summary.items():
        logger.info("  %-22s : %d", key, val)
    logger.info("─" * 50)


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_pipeline()
