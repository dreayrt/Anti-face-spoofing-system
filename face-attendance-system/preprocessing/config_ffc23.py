"""
config_ffc23.py – Cấu hình tập trung cho pipeline tiền xử lý FF-C23.
Tất cả đường dẫn, hằng số, và tham số đều định nghĩa ở đây.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# Đường dẫn
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "anti-spoof" / "raw" / "ff-c23" / "FaceForensics++_C23"
OUTPUT_DATA_DIR = PROJECT_ROOT / "dataset"
OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"

# Tạo thư mục output nếu chưa có
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Dataset name (dùng trong output path)
# ──────────────────────────────────────────────
DATASET_NAME = "ff-c23"

# ──────────────────────────────────────────────
# Các loại video
# ──────────────────────────────────────────────
REAL_DIR_NAME = "original"
SPOOF_METHODS = [
    "Deepfakes",
    "FaceSwap",
    "Face2Face",
    "NeuralTextures",
    "FaceShifter",
]

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
# Trích xuất frame
# ──────────────────────────────────────────────
FRAME_SAMPLE_RATE = 10
MAX_FRAMES_PER_VIDEO = 30
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

# ──────────────────────────────────────────────
# Cắt mặt (MTCNN)
# ──────────────────────────────────────────────
FACE_MARGIN = 40
MIN_FACE_SIZE = 40
FACE_DETECTION_THRESHOLD = [0.6, 0.7, 0.7]

# ──────────────────────────────────────────────
# Chia dữ liệu
# ──────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# ──────────────────────────────────────────────
# Augmentation
# ──────────────────────────────────────────────
ROTATION_DEGREES = 10
BRIGHTNESS_JITTER = 0.2
CONTRAST_JITTER = 0.2
GAUSSIAN_BLUR_KERNEL = 3
GAUSSIAN_BLUR_SIGMA = (0.1, 1.0)

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

# ──────────────────────────────────────────────
# Cleaning
# ──────────────────────────────────────────────
BLUR_THRESHOLD = 50.0
