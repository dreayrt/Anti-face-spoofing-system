"""
face_alignment.py – MTCNN-based face verification, alignment, and resizing.

Dùng cho SiW dataset (ảnh đã crop face sẵn):
  1. Verify: Xác nhận có khuôn mặt trong ảnh
  2. Align: Xoay mặt thẳng dựa trên landmarks 2 mắt (affine transform)
  3. Resize: Resize về IMAGE_SIZE × IMAGE_SIZE
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# MTCNN face detector (lazy init)
# ──────────────────────────────────────────────────────────────

_mtcnn = None


def _get_mtcnn(image_size: int = 224, margin: int = 30):
    """Lazy-load MTCNN detector."""
    global _mtcnn
    if _mtcnn is None:
        try:
            from facenet_pytorch import MTCNN
            _mtcnn = MTCNN(
                image_size=image_size,
                margin=margin,
                min_face_size=40,
                thresholds=[0.6, 0.7, 0.7],
                keep_all=False,
                post_process=False,
                device="cpu",
            )
            logger.info("MTCNN detector ready (face_alignment).")
        except ImportError:
            logger.error(
                "facenet-pytorch not found! "
                "Run: pip install facenet-pytorch"
            )
            raise
    return _mtcnn


# ──────────────────────────────────────────────────────────────
# Face Alignment sử dụng eye landmarks
# ──────────────────────────────────────────────────────────────


def align_face_from_landmarks(
    image: np.ndarray,
    left_eye: Tuple[float, float],
    right_eye: Tuple[float, float],
    output_size: int = 224,
) -> np.ndarray:
    """Align khuôn mặt dựa trên vị trí 2 mắt.

    Tính góc xoay giữa 2 mắt và thực hiện affine rotation
    để mắt nằm trên đường ngang.

    Parameters
    ----------
    image : np.ndarray
        Ảnh BGR (OpenCV format).
    left_eye : tuple (x, y)
        Tọa độ mắt trái.
    right_eye : tuple (x, y)
        Tọa độ mắt phải.
    output_size : int
        Kích thước output (output_size × output_size).

    Returns
    -------
    np.ndarray
        Ảnh đã aligned và resize.
    """
    # Tính góc xoay
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))

    # Tâm xoay = trung điểm giữa 2 mắt
    center_x = (left_eye[0] + right_eye[0]) / 2.0
    center_y = (left_eye[1] + right_eye[1]) / 2.0

    # Ma trận xoay
    h, w = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D(
        (center_x, center_y), angle, 1.0
    )
    rotated = cv2.warpAffine(
        image, rotation_matrix, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    # Resize về output_size
    resized = cv2.resize(rotated, (output_size, output_size), interpolation=cv2.INTER_AREA)
    return resized


# ──────────────────────────────────────────────────────────────
# Xử lý 1 ảnh: Verify + Align + Resize
# ──────────────────────────────────────────────────────────────


def process_single_image(
    image_path: Path,
    output_size: int = 224,
    margin: int = 30,
) -> Optional[np.ndarray]:
    """Xử lý 1 ảnh: detect face → align → resize.

    Nếu detect được face + landmarks → align + resize.
    Nếu detect được face nhưng không có landmarks → chỉ resize.
    Nếu không detect được face → vẫn resize (ảnh đã crop sẵn).

    Parameters
    ----------
    image_path : Path
        Đường dẫn tới file ảnh.
    output_size : int
        Kích thước output.
    margin : int
        Margin khi crop face.

    Returns
    -------
    np.ndarray or None
        Ảnh BGR đã xử lý, hoặc None nếu ảnh bị lỗi.
    """
    # Đọc ảnh
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        logger.debug("Cannot read image: %s", image_path.name)
        return None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # Detect face + landmarks bằng MTCNN
    mtcnn = _get_mtcnn(output_size, margin)
    try:
        boxes, probs, landmarks = mtcnn.detect(pil_img, landmarks=True)
    except Exception as exc:
        logger.debug("Detection error: %s (%s)", image_path.name, exc)
        # Fallback: chỉ resize
        return cv2.resize(img_bgr, (output_size, output_size), interpolation=cv2.INTER_AREA)

    # Nếu detect được face + landmarks
    if landmarks is not None and len(landmarks) > 0:
        # MTCNN trả về 5 landmarks: left_eye, right_eye, nose, mouth_left, mouth_right
        landmark = landmarks[0]  # Lấy face đầu tiên (lớn nhất)
        left_eye = (landmark[0][0], landmark[0][1])
        right_eye = (landmark[1][0], landmark[1][1])

        # Align + resize
        aligned = align_face_from_landmarks(
            img_bgr, left_eye, right_eye, output_size
        )
        return aligned

    # Nếu detect được box nhưng không có landmarks, hoặc không detect gì
    # → Ảnh đã crop sẵn, chỉ resize
    resized = cv2.resize(img_bgr, (output_size, output_size), interpolation=cv2.INTER_AREA)
    return resized


# ──────────────────────────────────────────────────────────────
# Batch processing
# ──────────────────────────────────────────────────────────────


def process_directory(
    input_dir: Path,
    output_dir: Path,
    output_size: int = 224,
    margin: int = 30,
) -> dict:
    """Xử lý toàn bộ ảnh trong thư mục: verify + align + resize.

    Parameters
    ----------
    input_dir : Path
        Thư mục chứa ảnh gốc.
    output_dir : Path
        Thư mục lưu ảnh đã xử lý.
    output_size : int
        Kích thước output.
    margin : int
        Margin cho MTCNN.

    Returns
    -------
    dict
        Thống kê: total, aligned, resized_only, failed.
    """
    from . import config_siw as config

    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS
    )

    stats = {"total": len(image_files), "aligned": 0, "resized_only": 0, "failed": 0}
    logger.info("  Processing %d images in '%s'...", len(image_files), input_dir.name)

    for i, img_path in enumerate(image_files):
        result = process_single_image(img_path, output_size, margin)

        if result is None:
            stats["failed"] += 1
            continue

        # Lưu ảnh đã xử lý
        save_path = output_dir / img_path.name
        cv2.imwrite(
            str(save_path), result,
            [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY],
        )

        # Kiểm tra xem ảnh có được align hay chỉ resize
        h, w = result.shape[:2]
        if h == output_size and w == output_size:
            stats["aligned"] += 1
        else:
            stats["resized_only"] += 1

        if (i + 1) % 200 == 0:
            logger.info(
                "    Processed %d/%d images...", i + 1, len(image_files)
            )

    logger.info(
        "  Done: %d aligned, %d resized-only, %d failed (total: %d)",
        stats["aligned"], stats["resized_only"], stats["failed"], stats["total"],
    )
    return stats
