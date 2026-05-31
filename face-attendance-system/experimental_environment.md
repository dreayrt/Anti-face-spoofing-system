# Thông tin Môi trường Thực nghiệm và Siêu tham số (Experimental Setup & Hyperparameters)

Tài liệu này cung cấp chi tiết về cấu hình phần cứng, phần mềm và các siêu tham số (Hyperparameters) được cấu hình trong quá trình huấn luyện (Training) mô hình chống giả mạo **CNN + DSP + LSTM**.

---

## 1. Cấu hình Phần cứng (Hardware Configuration)
*(Đã được trích xuất từ cấu hình máy tính cá nhân (Local) thực tế đang chạy)*

* **CPU:** 12th Gen Intel(R) Core(TM) i5-12450H
* **GPU (Card đồ họa):** NVIDIA GeForce RTX 4050 Laptop GPU (Sử dụng nhân CUDA để tăng tốc tính toán Deep Learning)
* **RAM (Bộ nhớ trong):** 16GB
* **Lưu trữ:** SSD GIGABYTE AG4512G-SI B10 (Đảm bảo tốc độ I/O nhanh khi load hàng ngàn ảnh từ Dataset)

## 2. Cấu hình Phần mềm (Software Configuration)
Hệ thống được phát triển và huấn luyện trên môi trường Python với các thư viện cốt lõi sau:

* **Ngôn ngữ lập trình:** Python 3.9 / 3.10
* **Nền tảng học sâu (Deep Learning Framework):** 
  * `torch >= 2.0.0` (PyTorch)
  * `torchvision >= 0.15.0`
* **Xử lý ảnh & Tính toán:**
  * `opencv-python >= 4.8.0` (Đọc và crop ảnh)
  * `Pillow >= 10.0.0`
  * `numpy >= 1.24.0`, `scipy >= 1.11.0` (Hỗ trợ xử lý tín hiệu DSP)
* **API Backend:**
  * `fastapi >= 0.104.0`, `uvicorn` (Triển khai Inference API)
* **Thư viện tối ưu tìm kiếm:**
  * `faiss` (Sử dụng HNSW để tìm kiếm vector 128D siêu tốc)

---

## 3. Các Siêu tham số Huấn luyện (Training Hyperparameters)
Các cấu hình này được trích xuất trực tiếp từ mã nguồn `train.py` để tối ưu hóa khả năng chống giả mạo.

### 3.1. Cấu hình Mô hình Cơ bản
* **Backbone (Mạng cơ sở):** `efficientnet_b0` (Được chọn vì cân bằng tốt giữa tốc độ và độ chính xác)
* **Kích thước ảnh đầu vào (Image Size):** `224 x 224` pixel
* **Kích thước Batch (Batch Size):** `32`
* **Số lượng Epochs:** `50`
* **Độ dài chuỗi khung hình (Sequence Length):** `5` frames (Dành cho luồng xử lý video đa khung hình - Multi-frame LSTM)

### 3.2. Thuật toán Tối ưu và Tốc độ học (Optimization & Learning Rate)
* **Optimizer:** `AdamW` (Tối ưu hóa có trọng số phạt)
* **Learning Rate (LR) ban đầu:** `5e-5` (0.00005)
* **Weight Decay (Phạt trọng số chống Overfitting):** `1e-4` (0.0001)
* **Scheduler (Điều chỉnh LR):** `CosineAnnealingLR` (Giảm LR theo hàm Cosine tới mức nhỏ nhất `1e-6`)
* **Gradient Clipping:** `1.0` (Tránh hiện tượng bùng nổ đạo hàm - Exploding Gradient)
* **Stochastic Weight Averaging (SWA):** Kích hoạt từ **Epoch 20** (Giúp làm mượt trọng số mô hình ở giai đoạn cuối, tăng khả năng tổng quát hóa).

### 3.3. Hàm mất mát (Loss Function)
Sử dụng **Focal Loss** thay vì Cross-Entropy tiêu chuẩn để xử lý vấn đề mất cân bằng dữ liệu và ép mô hình tập trung học các ca "khó" (Hard examples).
* **Gamma (Hệ số tập trung):** `2.0`
* **Spoof Weight (Hệ số phạt bất đối xứng - Alpha):** `2.5` (Phạt rất nặng nếu mô hình nhận diện nhầm ảnh giả thành người thật).
* **Label Smoothing:** `0.1` (Giảm độ "tự tin thái quá" của mô hình, giúp ranh giới phân loại mềm mại hơn).

### 3.4. Kỹ thuật Tăng cường Dữ liệu (Data Augmentation)
Để mô hình mạnh mẽ trước môi trường ánh sáng và góc quay khác nhau, các kỹ thuật sau được áp dụng (tỷ lệ kích hoạt ngẫu nhiên):
* **Mixup Alpha:** `0.4` (Trộn lẫn các bức ảnh và nhãn với nhau)
* **Dropout:** `0.6` tại lớp phân loại cuối cùng (Tránh học vẹt)
* **Biến đổi không gian & Màu sắc:** Horizontal Flip (50%), Rotation (15 độ), Color Jitter (sáng/tối/tương phản), Gaussian Blur.
* **Xóa/Che khuất vùng ảnh:** Random Erasing (30%) và Cutout (tạo lỗ thủng ngẫu nhiên trên ảnh).

### 3.5. Cơ chế Dừng sớm (Early Stopping)
* **Patience:** `5` Epochs. (Nếu Validation Loss không cải thiện sau 5 Epochs liên tiếp, quá trình train sẽ dừng sớm để tiết kiệm tài nguyên và tránh Overfitting).
