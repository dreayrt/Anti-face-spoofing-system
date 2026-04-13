# Preprocessing Pipeline – Changelog

> Ngày tạo: 2026-03-26

---

## 📦 Tổng Quan

Xây dựng pipeline tiền xử lý hoàn chỉnh cho bộ dữ liệu **CelebA Spoof Mini** (chống giả mạo khuôn mặt) bằng Python + PyTorch.

---

## 📁 Các File Đã Tạo

| File | Chức năng |
|---|---|
| `__init__.py` | Package init |
| `__main__.py` | Entry point (`python -m preprocessing`) |
| `config.py` | Hằng số, đường dẫn, tham số cấu hình |
| `cleaning.py` | Xóa ảnh hỏng, phát hiện trùng lặp (dhash), phát hiện khuôn mặt (Haar Cascade) |
| `splitting.py` | Chia dữ liệu train/val/test (70/15/15) có stratified |
| `augmentation.py` | Transform augmentation cho train & eval |
| `dataset.py` | PyTorch `Dataset`, `DataLoader`, `WeightedRandomSampler`, class weights |
| `visualization.py` | Grid ảnh augmented + biểu đồ phân bố lớp |
| `pipeline.py` | Script điều phối chính (6 bước tuần tự) |
| `requirements.txt` | Dependencies: torch, torchvision, Pillow, imagehash, opencv-python, scikit-learn, matplotlib, numpy |

---

## ✅ Tính Năng Đã Triển Khai

### 1. Data Cleaning
- Phát hiện & xóa ảnh hỏng (`PIL.Image.verify()` + `load()`)
- Phát hiện & xóa ảnh trùng lặp (perceptual hash – `dhash`)
- Phát hiện khuôn mặt bằng OpenCV Haar Cascade
- Logging chi tiết cho mọi bước

### 2. Data Transformation
- Resize tất cả ảnh → 224×224
- Normalize pixel values bằng ImageNet mean/std
- Convert sang tensor PyTorch

### 3. Xử Lý Mất Cân Bằng Dữ Liệu
- `get_class_weights()` — trọng số nghịch tần số cho `CrossEntropyLoss`
- `get_weighted_sampler()` — `WeightedRandomSampler` oversample lớp thiểu số
- In phân bố lớp trước và sau cân bằng

### 4. Chia Dữ Liệu
- Stratified random split: 70% train / 15% val / 15% test
- Ghi cảnh báo: không có nhãn identity → dùng random split (có thể data leakage nhẹ)

### 5. Data Augmentation (nhẹ nhàng, giữ đặc trưng spoof)
- `RandomHorizontalFlip` (p=0.5)
- `RandomRotation` (±10°)
- `ColorJitter` (brightness=0.2, contrast=0.2)
- `GaussianBlur` (kernel=3, sigma=0.1–1.0)

### 6. Output
- Thư mục `dataset/{train,val,test}/celeba-spoof/{live,spoof}/`
- PyTorch `Dataset` + `DataLoader` sẵn sàng dùng

### 7. Bonus
- Grid trực quan hóa mẫu augmented (`augmented_samples_*.png`)
- Biểu đồ phân bố lớp (`class_distribution.png`)
- Log toàn bộ pipeline (`pipeline.log`)

---

## 🔧 Thay Đổi Cấu Hình

| Tham số | Giá trị cũ | Giá trị mới | Lý do |
|---|---|---|---|
| `REMOVE_NO_FACE` | `False` | `True` | Xóa ảnh không phát hiện được khuôn mặt thay vì chỉ log |

---

## 🚀 Cách Chạy

```bash
# 1. Cài dependencies
cd d:\Deeplearning\face-attendance-system
pip install -r preprocessing/requirements.txt

# 2. Chạy pipeline
python -m preprocessing
```

---

## 📂 Output Kỳ Vọng

```
dataset/
├── train/celeba-spoof/live/
├── train/celeba-spoof/spoof/
├── val/celeba-spoof/live/
├── val/celeba-spoof/spoof/
├── test/celeba-spoof/live/
└── test/celeba-spoof/spoof/

preprocessing/outputs/
├── augmented_samples_train.png
├── augmented_samples_val.png
├── augmented_samples_test.png
├── class_distribution.png
└── pipeline.log
```
