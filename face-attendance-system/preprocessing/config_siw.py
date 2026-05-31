"""
config_siw.py – Cấu hình tập trung cho pipeline tiền xử lý SiW.
Tất cả đường dẫn, hằng số, và tham số đều định nghĩa ở đây.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# Đường dẫn
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "anti-spoof" / "raw" / "SiW"
OUTPUT_DATA_DIR = PROJECT_ROOT / "dataset"
OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"

# Tạo thư mục output nếu chưa có
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Dataset name (dùng trong output path)
# ──────────────────────────────────────────────
DATASET_NAME = "SiW"

# ──────────────────────────────────────────────
# Cấu trúc raw data
# SiW raw đã có sẵn split: train/val/test
# Mỗi split có 2 folder: real/ và spoof/
# ──────────────────────────────────────────────
RAW_CLASS_MAP = {
    "real": "live",    # raw folder → output folder
    "spoof": "spoof",
}

# ──────────────────────────────────────────────
# Cài đặt ảnh
# ──────────────────────────────────────────────
IMAGE_SIZE = 224
JPEG_QUALITY = 95
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ──────────────────────────────────────────────
# Face Alignment (MTCNN)
# ──────────────────────────────────────────────
FACE_MARGIN = 30           # Margin khi crop face (pixel)
MIN_FACE_SIZE = 40         # Kích thước mặt tối thiểu
FACE_DETECTION_THRESHOLD = [0.6, 0.7, 0.7]  # P-Net, R-Net, O-Net
ALIGNMENT_OUTPUT_SIZE = IMAGE_SIZE  # Kích thước output sau alignment

# ──────────────────────────────────────────────
# Augmentation (nhẹ, bảo toàn spoof artifacts)
# ──────────────────────────────────────────────
BRIGHTNESS_JITTER = 0.15   # Nhẹ
CONTRAST_JITTER = 0.15     # Nhẹ
GAUSSIAN_NOISE_STD = 0.01  # Gaussian noise nhẹ (σ)

# ──────────────────────────────────────────────
# Cleaning
# ──────────────────────────────────────────────
BLUR_THRESHOLD = 50.0      # Laplacian variance threshold

# ──────────────────────────────────────────────
# DataLoader
# ──────────────────────────────────────────────
BATCH_SIZE = 32
NUM_WORKERS = min(4, os.cpu_count() or 1)

# ──────────────────────────────────────────────
# Label
# ──────────────────────────────────────────────
CLASS_NAMES = ["live", "spoof"]
LIVE_LABEL = 0
SPOOF_LABEL = 1
