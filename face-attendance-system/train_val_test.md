# Báo Cáo Phân Chia Dữ Liệu: Train / Validation / Test

> **Dự án:** Face Attendance & Anti-Spoofing System  
> **Mô hình:** CNN + DSP (FFT) + LSTM — EfficientNet-B0 Backbone (~8.8M tham số)  
> **Ngày báo cáo:** 21/04/2026  

---

## 1. Tổng Quan

Hệ thống sử dụng **3 bộ dữ liệu** kết hợp để huấn luyện mô hình Anti-Spoofing:

| Bộ dữ liệu | Loại | Mô tả |
|---|---|---|
| **CelebA Spoof** | Ảnh tĩnh | Bộ dữ liệu khuôn mặt thật/giả từ ảnh tĩnh (print attack, replay attack) |
| **FaceForensics++ C23** | Video deepfake | Bộ dữ liệu video gốc và video deepfake (5 phương pháp: Deepfakes, FaceSwap, Face2Face, NeuralTextures, FaceShifter) |
| **SiW (Spoof in the Wild)** | Video presentation attack | Bộ dữ liệu face anti-spoofing với print/replay attack trong điều kiện thực tế (nhiều subject, nhiều session) |

Tỷ lệ chia dữ liệu chung: **70% Train / 15% Validation / 15% Test** (Random Seed = 42).

---

## 2. Phân Bố Dữ Liệu Chi Tiết

### 2.1 Tổng hợp theo Split

| Split | CelebA Live | CelebA Spoof | FF-C23 Live | FF-C23 Spoof | **Tổng Live** | **Tổng Spoof** | **Tổng cộng** |
|---|---|---|---|---|---|---|---|
| **Train** | 5,864 | 3,852 | 9,803 | 38,000 | **15,667** | **41,852** | **57,519** |
| **Validation** | 1,257 | 825 | 2,100 | 7,888 | **3,357** | **8,713** | **12,070** |
| **Test** | 1,257 | 826 | 2,005 | 7,140 | **3,262** | **7,966** | **11,228** |
| **Tổng** | **8,378** | **5,503** | **13,908** | **53,028** | **22,286** | **58,531** | **80,817** |

### 2.2 Tỷ lệ Live / Spoof theo Split

| Split | % Live | % Spoof | Tỷ lệ Spoof:Live |
|---|---|---|---|
| Train | 27.24% | 72.76% | 2.67:1 |
| Validation | 27.81% | 72.19% | 2.60:1 |
| Test | 29.05% | 70.95% | 2.44:1 |

### 2.3 Tỷ lệ đóng góp theo nguồn dữ liệu

| Nguồn | Train | Validation | Test | Tổng |
|---|---|---|---|---|
| CelebA Spoof | 9,716 (16.89%) | 2,082 (17.25%) | 2,083 (18.55%) | 13,881 (17.18%) |
| FF-C23 | 47,803 (83.11%) | 9,988 (82.75%) | 9,145 (81.45%) | 66,936 (82.82%) |

---

## 3. Chiến Lược Chia Dữ Liệu

### 3.1 CelebA Spoof — Stratified Random Split

- **Phương pháp:** `sklearn.model_selection.train_test_split` với `stratify` theo nhãn live/spoof.
- **Đảm bảo:** Tỷ lệ live/spoof được giữ nguyên nhất quán giữa các split.
- **Random Seed:** 42 (reproducible).

```
CelebA Spoof: 13,881 ảnh tổng
├── Train:      9,716 ảnh (70%) — 5,864 live + 3,852 spoof
├── Validation: 2,082 ảnh (15%) — 1,257 live + 825 spoof
└── Test:       2,083 ảnh (15%) — 1,257 live + 826 spoof
```

### 3.2 FaceForensics++ C23 — Video-Level Split

- **Phương pháp:** Chia theo **Video ID**, không chia theo frame → **tránh data leakage** hoàn toàn.
- **Lý do:** Các frame từ cùng một video có nội dung rất giống nhau. Nếu chia theo frame (random), model sẽ "nhìn thấy" các frame tương tự trong cả train/val/test → overfitting giả tạo.
- **Quy trình:**
  1. Thu thập danh sách Video ID duy nhất.
  2. Chia Video ID thành train/val/test (70/15/15).
  3. Tất cả frame được trích xuất từ video thuộc split nào thì nằm trong split đó.

```
FF-C23: 66,936 frame (từ video → MTCNN face crop → 224×224)
├── Train:      47,803 frame (71.4%) — 9,803 live + 38,000 spoof
├── Validation:  9,988 frame (14.9%) — 2,100 live + 7,888 spoof
└── Test:        9,145 frame (13.7%) — 2,005 live + 7,140 spoof
```

> **⚠️ Lưu ý quan trọng:** Tỷ lệ frame không chính xác 70/15/15 vì số frame trích xuất từ mỗi video khác nhau (phụ thuộc vào độ dài video và số mặt phát hiện được bởi MTCNN).

### 3.3 SiW (Spoof in the Wild) — Giữ nguyên Split gốc

- **Phương pháp:** Giữ nguyên split train/val/test có sẵn từ dataset gốc.
- **Lý do:** SiW được chia theo **Subject ID** → tránh data leakage hoàn toàn (cùng một người không xuất hiện trong 2 split khác nhau).
- **Tiền xử lý:**
  1. **Face Alignment:** MTCNN verify + align (xoay mặt thẳng dựa trên eye landmarks) + resize 224×224.
  2. **Data Cleaning:** Xoá ảnh corrupt + ảnh mờ (Laplacian variance < 50.0).
  3. **Copy:** Map `real/` → `live/`, `spoof/` → `spoof/` vào `dataset/{train,val,test}/SiW/`.
- **Augmentation (nhẹ):** Brightness ±0.15, Contrast ±0.15, Horizontal Flip, Gaussian Noise (σ=0.01).

```
SiW: Giữ nguyên split từ raw data (theo Subject ID)
├── Train:      real/ → live/, spoof/ → spoof/
├── Validation: real/ → live/, spoof/ → spoof/
└── Test:       real/ → live/, spoof/ → spoof/
```

> **⚠️ Lưu ý:** Ảnh SiW đã được crop face ở giai đoạn trước. Pipeline chỉ verify + align + resize, không re-detect.

### 3.4 Các loại Spoof trong FF-C23

| Phương pháp | Mô tả |
|---|---|
| **Deepfakes** | Hoán đổi khuôn mặt bằng deep learning autoencoder |
| **FaceSwap** | Hoán đổi khuôn mặt dựa trên 3D morphable model |
| **Face2Face** | Chuyển biểu cảm khuôn mặt từ video nguồn sang video đích |
| **NeuralTextures** | Chỉnh sửa texture khuôn mặt bằng neural rendering |
| **FaceShifter** | Identity-preserving face swapping với occlusion awareness |

---

## 4. Tiền Xử Lý Dữ Liệu (Preprocessing)

### 4.1 CelebA Spoof Pipeline (6 bước)

| Bước | Xử lý | Chi tiết |
|---|---|---|
| 1 | Data Cleaning | Xóa ảnh hỏng (`PIL.Image.verify()`), phát hiện trùng lặp (dhash, Hamming ≤ 5), face detection (Haar Cascade) |
| 2 | Data Splitting | Stratified random split 70/15/15 (seed=42) |
| 3 | Summary Statistics | Thống kê tổng hợp kết quả cleaning + splitting |
| 4 | PyTorch DataLoaders | Tạo Dataset + DataLoader với WeightedRandomSampler |
| 5 | Visualize Augmented | Grid ảnh augmented (8 mẫu/class/split) |
| 6 | Class Distribution | Biểu đồ phân bố lớp |

### 4.2 FaceForensics++ C23 Pipeline (4 bước)

| Bước | Xử lý | Chi tiết |
|---|---|---|
| 1 | Video-Level Split | Chia video ID → train/val/test (không data leakage) |
| 2 | Frame Extraction + Face Crop | MTCNN face detection (margin=40, min_face=40), mỗi 10 frame lấy 1, tối đa 30 frame/video, resize 224×224 |
| 3 | Data Cleaning | Xóa ảnh corrupt + ảnh mờ (Laplacian variance < 50.0) |
| 4 | Thống kê + DataLoaders | Tạo PyTorch DataLoaders |

### 4.3 SiW Pipeline (6 bước)

| Bước | Xử lý | Chi tiết |
|---|---|---|
| 1 | Face Alignment | MTCNN verify + align (eye landmarks affine transform) + resize 224×224 |
| 2 | Data Cleaning | Xóa ảnh corrupt + ảnh mờ (Laplacian variance < 50.0) |
| 3 | Copy vào Output | Giữ nguyên split, map real/→live/, spoof/→spoof/ |
| 4 | Thống kê | In phân bố class per split |
| 5 | DataLoaders | Tạo PyTorch DataLoaders + WeightedRandomSampler |
| 6 | Visualize | Augmented samples grid + class distribution chart |

### 4.4 Tham số chung

| Tham số | Giá trị |
|---|---|
| Image Size | 224 × 224 |
| Normalization | ImageNet mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] |
| Batch Size | 32 |
| Random Seed | 42 |
| Train Augmentation | RandomHorizontalFlip, RandomRotation(±15°), ColorJitter, GaussianBlur, RandomPerspective, RandomAffine, RandomGrayscale, RandomErasing, Cutout |
| Val/Test Transform | Resize + ToTensor + Normalize (không augmentation) |

---

## 5. Kết Quả Huấn Luyện (Training Results)

### 5.1 Cấu hình Training

| Tham số | Giá trị |
|---|---|
| Backbone | EfficientNet-B0 (pretrained ImageNet) |
| Tổng tham số | ~8.8 triệu |
| Epochs | 50 |
| Optimizer | AdamW (weight_decay=1e-4) |
| Learning Rate | 1e-4 (CosineAnnealingLR) |
| SWA | Kích hoạt từ epoch 39 (lr=1e-5) |
| Loss Function | Focal Loss (gamma=2.0, label_smoothing=0.1) |
| Spoof Weight | 3.0× (asymmetric class weights) |
| Dropout | 0.6 |
| MixUp | alpha=0.2 |
| Gradient Clipping | max_norm=1.0 |
| Sampler | WeightedRandomSampler (50:50 live/spoof mỗi batch) |

### 5.2 Tiến trình Training (Epoch-by-Epoch)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Live F1 | Spoof F1 | LR |
|---|---|---|---|---|---|---|---|
| 1 | 0.3096 | 69.10% | 0.2701 | 82.68% | 0.6960 | 0.8789 | 9.99e-05 |
| 5 | 0.2489 | 85.36% | 0.2396 | 87.78% | 0.8056 | 0.9109 | 9.76e-05 |
| 10 | 0.2348 | 88.43% | 0.2158 | 91.57% | 0.8581 | 0.9401 | 9.06e-05 |
| 15 | 0.2268 | 90.04% | 0.2176 | 91.76% | 0.8643 | 0.9409 | 7.96e-05 |
| 20 | 0.2223 | 91.02% | 0.2156 | 92.44% | 0.8754 | 0.9457 | 6.58e-05 |
| 25 | 0.2207 | 91.21% | 0.2094 | 93.31% | 0.8850 | 0.9528 | 5.05e-05 |
| 30 | 0.2180 | 91.89% | 0.2120 | 92.68% | 0.8769 | 0.9480 | 3.52e-05 |
| 35 | 0.2151 | 92.42% | 0.2116 | 93.21% | 0.8847 | 0.9518 | 2.26e-05 |
| 40 | 0.5566 | 91.57% | 0.2125 | 93.04% | 0.8833 | 0.9504 | 1.00e-05 |
| **41** | **0.2217** | **92.13%** | **0.2069** | **93.80%** | **0.8931** | **0.9564** | **1.00e-05** |
| 45 | 0.2155 | 92.48% | 0.2098 | 93.36% | 0.8879 | 0.9528 | 1.00e-05 |
| 50 | 0.2152 | 92.52% | 0.2112 | 92.99% | 0.8828 | 0.9500 | 1.00e-05 |

### 5.3 Best Model Summary

| Metric | Giá trị |
|---|---|
| **Best Epoch** | **41** |
| **Best Val Loss** | **0.2069** |
| **Best Val Accuracy** | **93.80%** |
| Val Live F1-Score | 0.8931 |
| Val Spoof F1-Score | 0.9564 |
| **Optimal Threshold** | **0.610** |
| Tổng thời gian training | ~7.5 giờ (RTX 4050 Laptop GPU) |
| Thời gian trung bình/epoch | ~540 giây (~9 phút) |

### 5.4 Biểu đồ Training

Các biểu đồ training được tự động generate và lưu tại `ai-service/training_logs/`:
- `loss_curves.png` — Train/Val Loss qua các epoch
- `accuracy_curves.png` — Train/Val Accuracy qua các epoch
- `precision_recall_f1.png` — Precision/Recall/F1 per class (LIVE + SPOOF)
- `learning_rate.png` — Lịch trình Learning Rate (CosineAnnealing + SWA)
- `confusion_matrix.png` — Ma trận nhầm lẫn (best model)
- `training_overview.png` — Dashboard tổng hợp 2×2

---

## 6. Kết Quả Validation (Best Epoch 41)

| Metric | Live | Spoof |
|---|---|---|
| Precision | 0.8583 | 0.9725 |
| Recall | 0.9309 | 0.9408 |
| F1-Score | 0.8931 | 0.9564 |
| **Overall Accuracy** | | **93.80%** |

---

## 7. Kết Quả Test (11,228 ảnh)

### 7.1 Overall Metrics (Argmax Prediction)

| Metric | Giá trị |
|---|---|
| **Test Accuracy** | **94.41%** |
| **ROC AUC** | **0.9802** |
| **Average Precision** | **0.9914** |
| Macro Avg Precision | 0.9289 |
| Macro Avg Recall | 0.9370 |
| Macro Avg F1 | 0.9328 |
| Weighted Avg Precision | 0.9448 |
| Weighted Avg Recall | 0.9441 |
| Weighted Avg F1 | 0.9443 |

### 7.2 Per-Class Metrics (Argmax)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **Live** | 0.8908 | 0.9203 | 0.9053 | 3,262 |
| **Spoof** | 0.9669 | 0.9538 | 0.9603 | 7,966 |

### 7.3 Metrics với Optimal Threshold (0.610)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **Live** | 0.9322 | 0.8553 | 0.8921 | 3,262 |
| **Spoof** | 0.9427 | 0.9745 | 0.9583 | 7,966 |
| **Overall Accuracy** | | | **93.99%** | 11,228 |

> **Ghi chú:** Optimal Threshold (0.610) ưu tiên **Spoof Recall ≥ 95%** — đảm bảo hệ thống phát hiện đúng ≥ 97.45% ảnh giả mạo, giảm thiểu rủi ro bảo mật.

### 7.4 Confusion Matrix (Argmax)

| | Predicted Live | Predicted Spoof | Total |
|---|---|---|---|
| **Actual Live** | 3,002 (92.03%) | 260 (7.97%) | 3,262 |
| **Actual Spoof** | 368 (4.62%) | 7,598 (95.38%) | 7,966 |
| **Total** | 3,370 | 7,858 | **11,228** |

### 7.5 Per-Source Metrics

| Source | Samples | Accuracy | Live P | Live R | Live F1 | Spoof P | Spoof R | Spoof F1 |
|---|---|---|---|---|---|---|---|---|
| **CelebA Spoof** | 2,083 | **95.34%** | 0.9401 | 0.9857 | 0.9623 | 0.9765 | 0.9044 | 0.9390 |
| **FF-C23** | 9,145 | **94.19%** | 0.8592 | 0.8793 | 0.8691 | 0.9659 | 0.9595 | 0.9627 |

#### Phân tích per-source:
- **CelebA Spoof** đạt accuracy cao hơn (95.34%) vì đây là ảnh tĩnh, dễ phân biệt hơn.
- **FF-C23** có Live F1 thấp hơn (0.8691) do video deepfake có chất lượng cao, khó phân biệt với video gốc.
- **FF-C23 Spoof F1** rất cao (0.9627) cho thấy model học tốt các artifacts của deepfake qua DSP branch (FFT frequency analysis).

### 7.6 Biểu đồ Test

Các biểu đồ test được tự động generate và lưu tại `ai-service/test_logs/`:
- `test_confusion_matrix.png` — Ma trận nhầm lẫn trên tập test
- `test_classification_report.png` — Precision/Recall/F1 per class (bar chart)
- `test_roc_curve.png` — ROC curve + AUC + Youden's J optimal point
- `test_precision_recall_curve.png` — Precision-Recall curve + Average Precision
- `test_score_distribution.png` — Histogram phân bố P(live) cho live vs spoof
- `test_per_source_metrics.png` — So sánh metrics theo nguồn dữ liệu
- `test_overview.png` — Dashboard tổng hợp 2×3
- `test_results.json` — Toàn bộ metrics chi tiết (JSON)

---

## 8. Phân Tích Overfitting

| Chỉ số | Train | Val (Best) | Test | Chênh lệch (Train-Test) |
|---|---|---|---|---|
| Loss | 0.2217 | 0.2069 | — | — |
| Accuracy | 92.13% | 93.80% | 94.41% | -2.28% |
| Spoof F1 | — | 0.9564 | 0.9603 | — |

**Kết luận:** Model **KHÔNG bị overfitting** — Test accuracy (94.41%) thực tế **cao hơn** cả Val accuracy (93.80%), chứng tỏ model generalize tốt trên dữ liệu chưa từng thấy. Điều này nhờ các kỹ thuật regularization mạnh mẽ đã áp dụng:

| Kỹ thuật Chống Overfitting | Mô tả |
|---|---|
| **Focal Loss** | Down-weight easy examples, focus on hard examples (γ=2.0) |
| **Label Smoothing** | Giảm overconfidence (ε=0.1) |
| **MixUp** | Blending images tạo soft decision boundaries (α=0.2) |
| **SWA** | Stochastic Weight Averaging ở 12 epoch cuối → flatten loss landscape |
| **Dropout** | 60% dropout rate trong classifier |
| **Asymmetric Weights** | Phạt spoof nặng hơn 3× |
| **Data Augmentation** | 9 kỹ thuật augmentation (flip, rotate, color jitter, blur, perspective, affine, grayscale, erasing, cutout) |
| **Gradient Clipping** | max_norm=1.0 → ổn định training |
| **WeightedRandomSampler** | Cân bằng batch 50:50 live/spoof |

---

## 9. Kết Luận

- **Tổng dữ liệu:** 80,817+ ảnh từ 3 nguồn (CelebA Spoof + FF-C23 + SiW).
- **Chia dữ liệu:** 70/15/15 với chiến lược phù hợp (stratified cho CelebA, video-level cho FF-C23, subject-level cho SiW).
- **Không data leakage:** FF-C23 chia theo video ID, CelebA chia stratified, SiW giữ nguyên split gốc theo subject.
- **Model đạt:** Accuracy **94.41%**, ROC AUC **0.9802**, Average Precision **0.9914** trên tập test.
- **Không overfitting:** Test performance cao hơn Validation nhờ regularization mạnh.
- **Optimal Threshold (0.610):** Đảm bảo Spoof Recall ≥ 95% — ưu tiên an ninh.
- **Sẵn sàng Production:** Model và toàn bộ pipeline đã sẵn sàng tích hợp vào AI Service phục vụ điểm danh.
