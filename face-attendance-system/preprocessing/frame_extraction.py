"""
frame_extraction.py – Trích xuất frame từ video, detect và crop khuôn mặt.

Dùng cho FF-C23:
  1. Đọc video bằng OpenCV VideoCapture
  2. Lấy mẫu mỗi N frame (FRAME_SAMPLE_RATE)
  3. Detect khuôn mặt bằng MTCNN
  4. Crop khuôn mặt lớn nhất + margin
  5. Resize về IMAGE_SIZE × IMAGE_SIZE
  6. Lưu thành JPEG
"""

import logging
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from . import config_ffc23 as config

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# MTCNN face detector (lazy init)
# ──────────────────────────────────────────────────────────────

_mtcnn = None


def _get_mtcnn():
    """Lazy-load MTCNN detector."""
    global _mtcnn
    if _mtcnn is None:
        try:
            from facenet_pytorch import MTCNN
            _mtcnn = MTCNN(
                image_size=config.IMAGE_SIZE,
                margin=config.FACE_MARGIN,
                min_face_size=config.MIN_FACE_SIZE,
                thresholds=config.FACE_DETECTION_THRESHOLD,
                keep_all=False,
                post_process=False,
                device="cpu",
            )
            logger.info("MTCNN detector đã sẵn sàng.")
        except ImportError:
            logger.error(
                "Không tìm thấy facenet-pytorch! "
                "Chạy: pip install facenet-pytorch"
            )
            raise
    return _mtcnn


# ──────────────────────────────────────────────────────────────
# Trích xuất frame từ 1 video
# ──────────────────────────────────────────────────────────────


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    video_stem: str,
    sample_rate: int = config.FRAME_SAMPLE_RATE,
    max_frames: int = config.MAX_FRAMES_PER_VIDEO,
) -> int:
    """Trích xuất frame + crop mặt từ 1 video. Trả về số frame đã lưu."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Không mở được video: %s", video_path)
        return 0

    mtcnn = _get_mtcnn()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_rate != 0:
            frame_idx += 1
            continue

        if saved_count >= max_frames:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            from PIL import Image
            pil_img = Image.fromarray(frame_rgb)
            face_img = mtcnn(pil_img)

            if face_img is not None:
                face_np = face_img.permute(1, 2, 0).numpy().astype(np.uint8)
                face_bgr = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)

                filename = f"{video_stem}_frame_{saved_count:04d}.jpg"
                save_path = output_dir / filename

                cv2.imwrite(
                    str(save_path),
                    face_bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY],
                )
                saved_count += 1
        except Exception as exc:
            logger.debug("Lỗi detect mặt frame %d của %s: %s", frame_idx, video_stem, exc)

        frame_idx += 1

    cap.release()
    return saved_count


# ──────────────────────────────────────────────────────────────
# Trích xuất cho 1 category
# ──────────────────────────────────────────────────────────────


def extract_category(
    raw_dir: Path,
    category: str,
    video_ids: List[str],
    output_dir: Path,
) -> int:
    """Trích xuất frame cho tất cả video trong 1 category. Trả về tổng frame."""
    category_dir = raw_dir / category
    total_saved = 0

    for i, vid_stem in enumerate(video_ids):
        video_path = None
        for ext in config.VIDEO_EXTENSIONS:
            candidate = category_dir / f"{vid_stem}{ext}"
            if candidate.is_file():
                video_path = candidate
                break

        if video_path is None:
            logger.warning("Không tìm thấy video: %s trong %s", vid_stem, category_dir)
            continue

        n = extract_frames_from_video(video_path, output_dir, vid_stem)
        total_saved += n

        if (i + 1) % 50 == 0 or (i + 1) == len(video_ids):
            logger.info(
                "  [%s] Đã xử lý %d/%d videos (%d frames)",
                category, i + 1, len(video_ids), total_saved,
            )

    return total_saved


# ──────────────────────────────────────────────────────────────
# Hàm chính: trích xuất tất cả
# ──────────────────────────────────────────────────────────────


def run_extraction(
    all_splits: Dict[str, Dict[str, List[str]]],
    raw_dir: Path | None = None,
    output_dir: Path | None = None,
) -> Dict[str, Dict[str, int]]:
    """Trích xuất frame cho toàn bộ dataset.

    Returns  { split_name: { "live": count, "Deepfakes": count, ... } }
    """
    raw_dir = raw_dir or config.RAW_DATA_DIR
    output_dir = output_dir or config.OUTPUT_DATA_DIR

    logger.info("=" * 60)
    logger.info("TRÍCH XUẤT FRAME + CẮT MẶT")
    logger.info("=" * 60)
    logger.info("Sample rate  : mỗi %d frame lấy 1", config.FRAME_SAMPLE_RATE)
    logger.info("Max frames   : %d frame/video", config.MAX_FRAMES_PER_VIDEO)
    logger.info("Image size   : %d×%d", config.IMAGE_SIZE, config.IMAGE_SIZE)

    stats: Dict[str, Dict[str, int]] = {}

    for split_name in ("train", "val", "test"):
        stats[split_name] = {}

        # Video thật → live/
        if config.REAL_DIR_NAME in all_splits:
            real_ids = all_splits[config.REAL_DIR_NAME].get(split_name, [])
            if real_ids:
                live_dir = output_dir / split_name / config.DATASET_NAME / "live"
                logger.info("")
                logger.info("▶ [%s] Trích xuất LIVE (%d videos)...", split_name, len(real_ids))
                n = extract_category(raw_dir, config.REAL_DIR_NAME, real_ids, live_dir)
                stats[split_name]["live"] = n
                logger.info("  → %d frames đã lưu", n)

        # Video giả → spoof/{method}/
        for method in config.SPOOF_METHODS:
            if method in all_splits:
                method_ids = all_splits[method].get(split_name, [])
                if method_ids:
                    spoof_dir = output_dir / split_name / config.DATASET_NAME / "spoof" / method
                    logger.info("")
                    logger.info("▶ [%s] Trích xuất %s (%d videos)...",
                                split_name, method, len(method_ids))
                    n = extract_category(raw_dir, method, method_ids, spoof_dir)
                    stats[split_name][method] = n
                    logger.info("  → %d frames đã lưu", n)

    return stats
