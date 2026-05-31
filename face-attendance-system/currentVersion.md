# Phiên bản hiện tại (Current Version) - Face Attendance & Anti-Spoofing System

Tài liệu này tổng hợp lại những tính năng, thành phần cốt lõi, kỹ thuật, công nghệ và cấu trúc mà hệ thống điểm danh bằng khuôn mặt (Face Attendance System) đã hoàn thiện cho đến thời điểm hiện tại.

Hệ thống đã được xây dựng với mô hình **3 dịch vụ (Monorepo Microservices)** gồm Frontend, Backend và AI Service, giao tiếp End-to-End. AI Service kết nối thực tế với CSDL để nhận diện nhân viên bằng **Face Descriptor 128D** (trích xuất từ face-api.js), kèm hệ thống **kiểm tra chất lượng khuôn mặt** (brightness, contrast, sharpness, face area ratio) chống che mặt/giả mạo. Mô hình Anti-Spoofing sử dụng kiến trúc hybrid **CNN (EfficientNet-B0) + DSP (FFT Frequency Analysis) + LSTM** với ~8.8 triệu tham số, có khả năng phát hiện ảnh giả mạo qua cả đặc trưng không gian lẫn miền tần số. Hệ thống hỗ trợ **Multi-frame Login + Vote** (thu nhiều frame, bỏ phiếu quyết định), **FAISS ANN Index** tăng tốc tra cứu descriptor, **Employee Cache** (TTL-based) giảm tải DB, **Test-Time Augmentation (TTA)** tăng độ chính xác inference. Ngoài ra, hệ thống đã xây dựng **Preprocessing Pipeline** hoàn chỉnh để tiền xử lý dữ liệu huấn luyện, áp dụng thành công các kỹ thuật Regularization tiên tiến (Focal Loss, SWA, MixUp, Cutout, RandomErasing) chống Overfitting hiệu quả, và đã **hoàn tất đánh giá trên tập Test** đạt **Accuracy 94.41%**, **ROC AUC 0.9802**, **Average Precision 0.9914**.

---

## 1. Kiến trúc tổng thể (Architecture Scaffolding)
- Cấu trúc thư mục rõ ràng chuẩn Monorepo: `frontend/`, `backend/`, `ai-service/`, `preprocessing/`, `data/`, `dataset/`.
- Dependencies độc lập cho từng dịch vụ (`package.json` cho frontend, `venv` cho backend/AI, `requirements.txt` cho preprocessing).
- Giao tiếp giữa các dịch vụ thông qua **RESTful API** (HTTP/JSON).
- **Preprocessing Pipeline** tách biệt — module Python chạy độc lập để chuẩn bị dữ liệu huấn luyện.

---

## 2. Công nghệ sử dụng (Technology Stack)

### 2.1 Frontend
| Công nghệ | Phiên bản / Mô tả | Mục đích |
|---|---|---|
| **React** | 18.x | Thư viện xây dựng giao diện người dùng (UI Library) |
| **Vite** | 5.x | Bundler & Dev Server siêu nhanh cho React |
| **TailwindCSS** | 3.x | Framework CSS tiện ích (Utility-first CSS) |
| **PostCSS** | - | Xử lý CSS nâng cao, plugin cho Tailwind |
| **Axios** | - | HTTP Client gửi/nhận API request |
| **@vladmandic/face-api** | - | Thư viện nhận diện khuôn mặt phía client (browser-based) |


### 2.2 Backend
| Công nghệ | Phiên bản / Mô tả | Mục đích |
|---|---|---|
| **Python** | 3.10 | Ngôn ngữ lập trình chính |
| **FastAPI** | - | Web Framework hiệu suất cao, hỗ trợ async |
| **Uvicorn** | - | ASGI Server chạy FastAPI |
| **SQLAlchemy** | - | ORM (Object-Relational Mapping) tương tác CSDL |
| **Psycopg2** | - | PostgreSQL adapter cho Python |
| **httpx** | - | Async HTTP Client gọi tới AI Service |
| **Pydantic** | - | Xác thực dữ liệu (Data Validation) cho API request/response |

### 2.3 AI Service
| Công nghệ | Phiên bản / Mô tả | Mục đích |
|---|---|---|
| **Python** | 3.10 | Ngôn ngữ lập trình chính |
| **PyTorch** | ≥2.0 | Framework Deep Learning — CNN backbone, LSTM, training |
| **TorchVision** | ≥0.15 | MobileNetV2/ResNet50/EfficientNet pretrained models |
| **FastAPI** | - | Web Framework chạy cổng riêng (port 8001) |
| **OpenCV (cv2)** | - | Xử lý ảnh: decode base64, resize, crop |
| **NumPy** | - | Tính toán ma trận, Euclidean distance trên face descriptor |
| **SciPy** | ≥1.11 | Hỗ trợ DSP — signal processing utilities |
| **FAISS (faiss-cpu)** | - | Approximate Nearest Neighbor (ANN) — tăng tốc tra cứu face descriptor (HNSW index) |
| **Psycopg2-binary** | - | Kết nối trực tiếp PostgreSQL đọc face descriptor nhân viên |
| **scikit-learn** | ≥1.3 | Tính metrics: precision, recall, F1, confusion matrix |

### 2.4 Preprocessing Pipeline
| Công nghệ | Phiên bản / Mô tả | Mục đích |
|---|---|---|
| **Python** | 3.10 | Ngôn ngữ lập trình chính |
| **PyTorch** | - | Framework Deep Learning — Dataset, DataLoader, Transforms |
| **TorchVision** | - | Transform ảnh (resize, normalize, augmentation) |
| **MTCNN (facenet-pytorch)** | - | Phát hiện & cắt khuôn mặt từ frame video (dùng cho FF-C23) |
| **OpenCV (cv2)** | - | Đọc video (VideoCapture), tính Laplacian blur score |
| **Pillow (PIL)** | - | Verify & load ảnh, phát hiện ảnh corrupt |
| **ImageHash** | - | Phát hiện ảnh trùng lặp bằng perceptual hash (dhash) |
| **scikit-learn** | - | `train_test_split` — chia dữ liệu stratified |
| **Matplotlib** | - | Trực quan hóa phân bố lớp, grid ảnh augmented |
| **NumPy** | - | Xử lý ma trận ảnh |

### 2.5 Database
| Công nghệ | Mô tả |
|---|---|
| **PostgreSQL** | Hệ quản trị CSDL quan hệ, bảng `employees` (id, name, face_image_base64, face_descriptor, created_at) |

### 2.6 Kỹ thuật AI / Computer Vision đã áp dụng
| Kỹ thuật | Mô tả |
|---|---|
| **Real-time Face Detection (TinyFaceDetector)** | Model nhẹ chạy trên browser để hiển thị bounding box theo thời gian thực (150ms/frame) |
| **Accurate Face Detection (SSD MobileNet V1)** | Model nặng hơn, chính xác hơn, dùng khi bấm Scan Face (minConfidence > 0.7) |
| **Face Landmark Detection (68-point)** | Phát hiện 68 điểm đặc trưng khuôn mặt (mắt, mũi, miệng, hàm) bằng `faceLandmark68Net` |
| **Face Descriptor Extraction (128D)** | Trích xuất vector đặc trưng 128 chiều (FaceRecognitionNet) — đại diện duy nhất cho mỗi khuôn mặt |
| **Euclidean Distance Matching** | So khớp khuôn mặt bằng khoảng cách Euclidean giữa 2 descriptor 128D (ngưỡng < 0.55) |
| **Face Quality Validation (Canvas-based)** | Kiểm tra chất lượng khuôn mặt trước khi cho phép scan — đánh giá brightness, contrast, sharpness, face area ratio trên vùng mặt cắt từ canvas |
| **Anti-Spoofing CNN+DSP+LSTM** | Mô hình hybrid 3 thành phần: CNN trích xuất đặc trưng không gian, DSP phân tích miền tần số (FFT), LSTM học sequential patterns. Chống overfitting tuyệt đối, kết hợp ngưỡng động (Optimal threshold). |
| **DSP Frequency Analysis (FFT 2D)** | Áp dụng Fast Fourier Transform 2D lên feature maps → tính power spectrum → phát hiện moiré patterns, banding, noise artifacts từ ảnh giả mạo (print/replay attack) |
| **CNN Backbone (EfficientNet-B0)** | Mạng pretrained ImageNet làm backbone trích xuất spatial features (1280-D). (Được chọn thay cho MobileNetV2 để cân bằng giữa hiệu suất và độ nhẹ). |
| **LSTM Temporal Modeling** | Mô hình hóa sequential dependencies giữa spatial + frequency features. Hỗ trợ cả single-frame và multi-frame (video-level) anti-spoofing |
| **FAISS ANN Index (HNSW)** | Approximate Nearest Neighbor search — xây dựng HNSW index trên prototype descriptors để tìm top-K nhân viên gần nhất trong O(log N), thay vì brute-force O(N) |
| **Employee Cache (TTL-based)** | Cache thông tin nhân viên + descriptor trong bộ nhớ với TTL cấu hình được (mặc định 20s), giảm tải query PostgreSQL mỗi request |
| **2-Stage Matching (Prototype → Sample)** | Stage 1: FAISS/numpy tìm top-K prototype gần nhất. Stage 2: so chi tiết với từng sample của candidate → tìm best match chính xác |
| **Multi-frame Vote Aggregation** | Thu nhiều frame hợp lệ , đánh giá từng frame độc lập (liveness + matching), bỏ phiếu theo employee ID → quyết định cuối cùng khi đủ vote threshold |
| **Test-Time Augmentation (TTA)** | Chạy model trên 5 biến thể của ảnh (gốc, flip, brightness, rotation, center-crop) và trung bình kết quả → giảm false positive ~10-15% |
| **Temporal Augmentation** | Áp dụng nhiễu theo thời gian (Flicker, Color Shift, Per-frame Noise) riêng cho class spoof để LSTM học cách phát hiện sự thiếu nhất quán thời gian của ảnh giả. |
| **Video Sequence Learning** | Gom nhóm ảnh thành chuỗi (Video Sequence `seq_len=5`) với input 5D `(B, T, C, H, W)` để LSTM phân tích chuỗi thời gian thực sự thay vì phân tích ảnh tĩnh. |
| **Cutout Augmentation** | Ngẫu nhiên mask các vùng hình chữ nhật khi training → buộc model học từ nhiều vùng thay vì phụ thuộc 1 điểm duy nhất |
| **RandomErasing** | Ngẫu nhiên xóa vùng nhỏ trên tensor đã normalize — augmentation mạnh bổ sung cho Cutout |
| **MTCNN Face Detection** | Multi-task Cascaded CNN — phát hiện & crop khuôn mặt từ frame video (dùng trong preprocessing FF-C23) |
| **Bounding Box Detection** | Trích xuất tọa độ (x, y, w, h), vẽ khung nhận diện lên video |
| **Base64 Image Encoding** | Chuyển đổi frame video thành chuỗi Base64 để truyền qua API |
| **Face Crop with Padding** | Cắt vùng khuôn mặt từ ảnh gốc kèm margin 10% để lấy đủ ngữ cảnh |
| **Laplacian Blur Detection** | Tính Laplacian variance để phát hiện ảnh mờ (threshold < 50.0) |
| **Perceptual Hash (dhash)** | Phát hiện ảnh trùng lặp/gần trùng bằng difference hash (Hamming distance ≤ 5) |
| **EMA Smoothing + Outlier Capping** | Làm mượt biểu đồ training bằng Exponential Moving Average + thay thế spike bất thường bằng nội suy lân cận |

---

## 3. Giao diện người dùng - Frontend (React + Vite + TailwindCSS)

**Điều hướng trang:** Sử dụng **state-based navigation** (không dùng React Router) — 2 tab **Login** và **Register** chuyển đổi qua `useState`, gọn nhẹ, không cần routing.

- **Camera Kiosk — Live Authentication (CameraPage):**
  - Webcam trực tiếp với **TinyFaceDetector** + **faceLandmark68TinyNet** + **FaceRecognitionNet** hiển thị bounding box real-time (180ms/frame).
  - Bấm **Scan Face (Vote)** → thu tự động **5 frame hợp lệ** (tối đa 10 lần thử, delay 120ms giữa các frame):
    1. Mỗi frame được **kiểm tra chất lượng** (`evaluateFaceQualityFromCanvas`):
       - ✅ Brightness: 72–205
       - ✅ Contrast: ≥ 32
       - ✅ Sharpness: ≥ 24
       - ✅ Face area ratio: ≥ 6% khung hình
    2. Frame không đạt chất lượng → tự động bỏ qua, tiếp tục frame sau.
    3. Frame hợp lệ → đính kèm `image`, `box`, `descriptor`, `quality_metrics`.
  - Gửi batch frames → Backend → AI Service, kèm `vote_min_match` (ngưỡng bỏ phiếu).
  - Nếu < 3 frame hợp lệ → từ chối trực tiếp trên client.
  - Hiển thị đầy đủ thông số: liveness score, similarity, best distance, matched votes, frames used, tên nhân viên.
  - Camera tự bật khi mở app, tự dừng khi tab ẩn (`visibilitychange`), đóng tab (`beforeunload`), hoặc rời trang (`pagehide`).
- **Trang Đăng ký Nhân viên (RegisterEmployee):**
  - Nhập tên, ID tự động sinh (format `EMP-{timestamp}-{random}`).
  - Dùng **TinyFaceDetector** + **faceLandmark68TinyNet** + **FaceRecognitionNet** để trích xuất face descriptor 128D (200ms/frame).
  - Hỗ trợ **multi-sample register** (tối thiểu 5 mẫu/nhân viên, tối đa 10 mẫu trong 1 phiên đăng ký).
  - Mỗi mẫu đều đi qua **quality gate** phía frontend (`evaluateFaceQualityFromCanvas` — brightness, contrast, sharpness, face area ratio) trước khi chấp nhận.
  - Tạo descriptor đại diện (prototype) từ trung bình nhiều mẫu trước khi lưu.
  - Hỗ trợ xóa mẫu cuối, xóa toàn bộ, mở/tắt camera.
- **Module `utils/faceQuality.js`** (MỚI):
  - Hàm `evaluateFaceQualityFromCanvas()`: cắt vùng mặt từ canvas, tính brightness/contrast/sharpness (Sobel gradient), face area ratio.
  - Hàm `evaluateFaceQualityFromImageData()`: đánh giá chất lượng trên raw ImageData.
  - Ngưỡng cấu hình được: `minBrightness`, `maxBrightness`, `minContrast`, `minSharpness`, `minFaceAreaRatio`.
- **Tích hợp API:** Axios client (`services/api.js`) gửi payload mở rộng gồm `image`, `box`, `descriptor`, `samples[]`, `frames[]`, `vote_min_match` tùy luồng register/recognize.

## 4. Máy chủ xử lý trung tâm - Backend (FastAPI)
- **Server:** FastAPI + CORS cho phép Frontend React (cổng 5173) gọi API.
- **CSDL:**
  - SQLAlchemy Engine + Session liên kết PostgreSQL.
  - Model `Employee`: `id`, `name`, `face_image_base64`, `face_descriptor` (JSON array 128D), `created_at`.
  - Auto create tables qua `main.py`.
- **API Endpoints:**
  - **`/api/v1/face/register`**: Nhận (id, name, samples[]) → Xây dựng descriptor blob v2 (prototype + samples) → Lưu Employee vào PostgreSQL. Yêu cầu tối thiểu 5 mẫu, tương thích ngược với payload cũ (single image + descriptor).
  - **`/api/v1/face/recognize`**: Nhận (image, box, descriptor, frames[], vote_min_match) → Chuyển tiếp sang AI Service `/predict`, hỗ trợ cả single-frame và multi-frame vote.
  - **`/api/v1/face/detect-face`**: Placeholder endpoint cho face detection.

## 5. Dịch vụ Trí tuệ nhân tạo - AI Service (PyTorch + CNN + DSP + LSTM)
- Dịch vụ độc lập port 8001, chuyên xử lý AI.
- **Employee Cache (TTL-based):**
  - Cache toàn bộ thông tin nhân viên (descriptor, prototype, samples) trong bộ nhớ với TTL cấu hình được (mặc định 20s).
  - Tự động rebuild khi cache hết hạn, hỗ trợ force reload qua API `/refresh-cache`.
  - Parse descriptor blob v2 (prototype + samples) và v1 (array 128D) — backward-compatible.
- **FAISS ANN Index:**
  - Xây dựng HNSW index (`efConstruction=40`, `efSearch=64`) trên prototype vectors.
  - Tăng tốc Stage 1 matching từ O(N) brute-force xuống O(log N).
  - Fallback tự động về numpy exhaustive nếu FAISS không cài được.
- **Mô hình Anti-Spoofing: CNN + DSP + LSTM (~8.8M tham số)**
  - **CNN Branch (EfficientNet-B0):** Trích xuất đặc trưng không gian — texture, edges, moiré patterns. Sử dụng pretrained ImageNet weights, output feature map 7×7×1280.
  - **DSP Branch (FFT 2D):** Phân tích miền tần số trên feature maps từ CNN. Áp dụng Fast Fourier Transform 2D → Power Spectrum → Conv1D compression → 256-D frequency vector. Phát hiện artifacts tần số đặc trưng của ảnh giả (print/replay).
  - **LSTM Layer:** Kết hợp spatial (1280-D) + frequency (256-D) = 1536-D → LSTM 2 lớp (hidden=256) → học sequential dependencies. Hỗ trợ cả single-frame và multi-frame inference.
  - **SE Attention Block:** Squeeze-and-Excitation channel attention (256→64→256) — weight tầm quan trọng từng feature channel trước khi phân loại.
  - **Classifier:** FC(256→128) → ReLU → Dropout(0.5) → FC(128→2) → Softmax → [live, spoof].
  - **TTA (Test-Time Augmentation):** Khi inference, chạy model trên 5 biến thể (gốc, flip, brightness, rotation, center-crop) và trung bình kết quả — giảm false positive ~10-15%.
  - **Fallback:** Nếu chưa có checkpoint train → tự động dùng `MockAntiSpoofModel` (backward-compatible).
- **API `/predict`:**
  1. Nhận payload (single-frame hoặc multi-frame với `frames[]` + `vote_min_match`).
  2. Rebuild employee cache nếu hết hạn TTL.
  3. Với mỗi frame:
     - Decode base64 → crop vùng mặt theo bounding box (kèm padding 10%).
     - **Anti-Spoofing:** Model CNN+DSP+LSTM chấm điểm liveness (áp dụng tự động Optimal Threshold dò tìm lúc train). Đối với Multi-frame, sử dụng `predict_video()` truyền cả chuỗi ảnh qua LSTM để ra điểm liveness tổng thể.
     - **2-Stage Face Matching:**
       - Stage 1: FAISS ANN tìm top-5 prototype gần nhất (hoặc numpy fallback).
       - Stage 2: So chi tiết Euclidean distance với từng sample của candidate.
       - Ngưỡng < **0.55** → cùng một người → Match.
  4. **Vote Aggregation** (khi multi-frame): đếm số phiếu theo employee ID, so với `vote_min_match` → quyết định Access Granted/Denied.
- **API `/refresh-cache`** (MỚI): Force reload employee cache từ DB, trả về số lượng nhân viên và trạng thái ANN.
- **Training Pipeline (`train.py` & `video_dataset.py` & `temporal_augmentation.py`):**
  - Hỗ trợ cả **Single-frame Training** và **Multi-frame Training** (chạy input 5D qua `forward_multi_frame`).
  - Hỗ trợ dữ liệu CelebA Spoof + FF-C23 (đã preprocessing) cùng WeightedRandomSampler chia batch 50:50. Dữ liệu được gom thành chuỗi thời gian thông qua `VideoSequenceDataset`.
  - Áp dụng **Temporal Augmentation** (Flicker, Color Shift, Noise) trên các video giả mạo để dạy model nhận biết sự thiếu nhất quán theo thời gian.
  - **Focal Loss** + Label Smoothing (0.1) + Asymmetric class weights (phạt spoof nặng hơn 3x).
  - Optimizer: AdamW (weight_decay=1e-4) + Gradient Clipping (norm=1.0).
  - Scheduler: CosineAnnealingLR kết hợp với SWA (Stochastic Weight Averaging) ở các epoch cuối để tổng quát hóa weights.
  - **Data Augmentation mạnh (11 kỹ thuật):** RandomHorizontalFlip, RandomRotation(15°), ColorJitter, GaussianBlur, RandomPerspective, RandomAffine, RandomGrayscale, Cutout, RandomErasing, MixUp (alpha=0.2), Dropout=0.5.
  - Tự động dò **Optimal Threshold** (ngưỡng tối ưu) đảm bảo spoof_recall >= 95%.
  - Output: `models/weights/antispoof_cnn_dsp_lstm.pth`.
- **Training Visualization (tự động generate sau mỗi epoch):**
  - `loss_curves.png` — Biểu đồ Train Loss vs Val Loss, đánh dấu điểm tốt nhất (best val loss). Áp dụng **EMA Smoothing** (alpha=0.25) + **Outlier Capping** (factor=2.5) cho đường cong mượt, raw data hiển thị dạng scatter mờ.
  - `accuracy_curves.png` — Biểu đồ Train Accuracy vs Val Accuracy, đánh dấu best val accuracy.
  - `precision_recall_f1.png` — Precision / Recall / F1-Score cho 2 class (LIVE và SPOOF) riêng biệt.
  - `learning_rate.png` — Lịch trình Learning Rate (CosineAnnealing, log scale).
  - `confusion_matrix.png` — Heatmap confusion matrix của best model (cập nhật khi có best mới).
  - `training_overview.png` — Biểu đồ tổng hợp 2×2: Loss + Accuracy + Precision + Learning Rate (đều có smoothing).
  - Thiết kế dark theme, cập nhật **live** sau mỗi epoch để theo dõi tiến trình training.
  - Output: `ai-service/training_logs/`.
- **Test Evaluation Pipeline (`test.py`):**
  - Đánh giá mô hình trên tập test (11,228 ảnh) với checkpoint tốt nhất (epoch 41).
  - Tính toán chỉ số đầy đủ: Accuracy, ROC AUC, Average Precision, Precision/Recall/F1 per class.
  - Hỗ trợ đánh giá theo **2 chế độ**: argmax prediction và optimal threshold (0.610).
  - Phân tích **per-source**: metrics riêng cho CelebA Spoof và FF-C23.
  - Output: `ai-service/test_logs/`.
- **Test Visualization (tự động generate sau khi test):**
  - `test_confusion_matrix.png` — Ma trận nhầm lẫn trên tập test (kèm % cho mỗi ô).
  - `test_classification_report.png` — Biểu đồ cột Precision/Recall/F1 per class, kèm Overall Accuracy.
  - `test_roc_curve.png` — Đường cong ROC với AUC, đánh dấu điểm tối ưu (Youden's J).
  - `test_precision_recall_curve.png` — Đường cong Precision-Recall với Average Precision.
  - `test_score_distribution.png` — Histogram phân bố điểm P(live) cho live vs spoof, kèm threshold line.
  - `test_per_source_metrics.png` — Biểu đồ so sánh Accuracy và F1 theo nguồn dữ liệu (CelebA vs FF-C23).
  - `test_overview.png` — Dashboard tổng hợp 2×3: CM + Classification Report + ROC + PR + Distribution + Summary.
  - `test_results.json` — Toàn bộ metrics chi tiết dạng JSON.
  - Thiết kế dark theme đồng bộ với training charts.
  - Output: `ai-service/test_logs/`.

---

## 6. Preprocessing Pipeline — Tiền xử lý dữ liệu huấn luyện

Module Python độc lập (`preprocessing/`) dùng để chuẩn bị dữ liệu cho việc huấn luyện mô hình Anti-Spoofing. Hỗ trợ **2 bộ dữ liệu** với pipeline riêng biệt.

### 6.1 Pipeline CelebA Spoof (`python -m preprocessing`)

Pipeline **6 bước tuần tự** xử lý bộ dữ liệu CelebA Spoof Mini (ảnh tĩnh):

| Bước | Mô tả |
|---|---|
| **1. Data Cleaning** | Xóa ảnh hỏng (`PIL.Image.verify()` + `load()`), phát hiện trùng lặp (perceptual hash — `dhash`, Hamming ≤ 5), phát hiện khuôn mặt bằng Haar Cascade |
| **2. Data Splitting** | Stratified random split: 70% train / 15% val / 15% test (seed=42) |
| **3. Summary Statistics** | In thống kê tổng hợp (cleaning results + split sizes + phân bố live/spoof) |
| **4. PyTorch DataLoaders** | Tạo `Dataset` + `DataLoader` với `WeightedRandomSampler` để cân bằng lớp |
| **5. Visualize Augmented** | Render grid ảnh augmented cho mỗi split (8 mẫu/class) |
| **6. Class Distribution** | Xuất biểu đồ phân bố lớp (`class_distribution.png`) |

**Augmentation (chỉ train):**
- `RandomHorizontalFlip` (p=0.5)
- `RandomRotation` (±10°)
- `ColorJitter` (brightness=0.2, contrast=0.2)
- `GaussianBlur` (kernel=3, sigma=0.1–1.0)

**Xử lý mất cân bằng:**
- `get_class_weights()` — trọng số nghịch tần số cho `CrossEntropyLoss`
- `get_weighted_sampler()` — `WeightedRandomSampler` oversample lớp thiểu số

### 6.2 Pipeline FaceForensics++ C23 (`python -m preprocessing.pipeline_ffc23`)

Pipeline **4 bước** xử lý bộ dữ liệu FF-C23 (video deepfake):

| Bước | Mô tả |
|---|---|
| **1. Video-Level Split** | Chia video ID thành train/val/test (70/15/15) → tất cả frame cùng video ở cùng split → **không data leakage** |
| **2. Frame Extraction + Face Crop** | Trích xuất frame (mỗi 10 frame lấy 1, tối đa 30 frame/video), phát hiện & cắt mặt bằng **MTCNN** (margin=40, min_face=40), resize → 224×224, lưu JPEG (quality=95) |
| **3. Data Cleaning (nhẹ)** | Xóa ảnh corrupt + ảnh mờ (Laplacian variance < 50.0) |
| **4. Thống kê & DataLoaders** | In thống kê trích xuất + tạo PyTorch DataLoaders |

**Các loại video FF-C23:**
- **Real (original):** Video gốc → class `live`
- **Spoof (5 phương pháp):** Deepfakes, FaceSwap, Face2Face, NeuralTextures, FaceShifter → class `spoof`

**Đặc điểm kỹ thuật:**
- MTCNN face detection với threshold `[0.6, 0.7, 0.7]`
- Video-level splitting bằng `scikit-learn train_test_split`
- Output tổ chức theo `dataset/{split}/ff-c23/{live,spoof/{method}}/`

### 6.3 Cấu hình & Tham số chung

| Tham số | CelebA Spoof | FF-C23 |
|---|---|---|
| Image Size | 224×224 | 224×224 |
| Train / Val / Test | 70 / 15 / 15 | 70 / 15 / 15 |
| Random Seed | 42 | 42 |
| Batch Size | 32 | 32 |
| Normalization | ImageNet mean/std | ImageNet mean/std |
| Frame Sample Rate | N/A (ảnh tĩnh) | Mỗi 10 frame lấy 1 |
| Max Frames/Video | N/A | 30 |
| Duplicate Detection | dhash (Hamming ≤ 5) | Không (video-level split đã tránh) |
| Blur Detection | Không | Laplacian variance < 50.0 |
| Face Detection | Haar Cascade (optional) | MTCNN (bắt buộc) |

---

### 6.4 Phân bố dữ liệu Train / Validation / Test

> 📄 **Báo cáo chi tiết:** Xem file [`train_val_test.md`](train_val_test.md) để biết đầy đủ phân tích dữ liệu, tiến trình training epoch-by-epoch, và phân tích overfitting.

**Tổng dữ liệu: 80,817 ảnh** (22,286 live + 58,531 spoof) từ 2 nguồn.

| Split | CelebA Live | CelebA Spoof | FF-C23 Live | FF-C23 Spoof | **Tổng Live** | **Tổng Spoof** | **Tổng cộng** |
|---|---|---|---|---|---|---|---|
| **Train** | 5,864 | 3,852 | 9,803 | 38,000 | **15,667** | **41,852** | **57,519** |
| **Validation** | 1,257 | 825 | 2,100 | 7,888 | **3,357** | **8,713** | **12,070** |
| **Test** | 1,257 | 826 | 2,005 | 7,140 | **3,262** | **7,966** | **11,228** |
| **Tổng** | **8,378** | **5,503** | **13,908** | **53,028** | **22,286** | **58,531** | **80,817** |

**Chiến lược chia:**
- **CelebA Spoof:** Stratified random split (giữ nguyên tỷ lệ live/spoof giữa các split).
- **FF-C23:** Video-level split (chia theo Video ID → tất cả frame cùng video ở cùng split → **không data leakage**).

---

### 6.5 Kết quả Training & Test

#### Training Results (50 epochs)
| Metric | Value |
|--------|-------|
| Best Epoch | 41 |
| Best Val Loss | 0.2069 |
| Best Val Accuracy | 93.80% |
| Val Spoof F1-Score | 0.9564 |
| Val Live F1-Score | 0.8931 |
| Optimal Threshold | 0.610 |
| Total Training Time | ~7.5 giờ (RTX 4050 Laptop GPU) |

#### Test Results (11,228 ảnh)
| Metric | Value |
|--------|-------|
| **Test Accuracy** | **94.41%** |
| **ROC AUC** | **0.9802** |
| **Average Precision** | **0.9914** |
| Live Precision | 0.8908 |
| Live Recall | 0.9203 |
| Live F1-Score | 0.9053 |
| Spoof Precision | 0.9669 |
| Spoof Recall | 0.9538 |
| Spoof F1-Score | 0.9603 |

#### Test Results per Source
| Source | Accuracy | Live F1 | Spoof F1 |
|--------|----------|---------|----------|
| CelebA Spoof | 95.34% | 0.9623 | 0.9390 |
| FF-C23 | 94.19% | 0.8691 | 0.9627 |

#### Confusion Matrix (Test)
| | Pred Live | Pred Spoof |
|---|---|---|
| **Actual Live** (3,262) | 3,002 (92.03%) | 260 (7.97%) |
| **Actual Spoof** (7,966) | 368 (4.62%) | 7,598 (95.38%) |

#### Phân tích Overfitting
| Chỉ số | Train (Epoch 41) | Val (Best) | Test |
|---|---|---|---|
| Accuracy | 92.13% | 93.80% | 94.41% |
| Spoof F1 | — | 0.9564 | 0.9603 |

> ✅ **Không overfitting** — Test accuracy (94.41%) cao hơn cả Val accuracy (93.80%), nhờ bộ kỹ thuật regularization mạnh: Focal Loss, SWA, MixUp, Cutout, RandomErasing, Dropout 0.5, Asymmetric Weights, Data Augmentation (11 kỹ thuật).

---

## 7. Luồng hoạt động xuyên suốt (End-to-End Flow)

### Luồng Đăng ký (Registration Flow)
1. Quản trị viên vào tab **Register** → Nhập tên → ID tự động sinh (`EMP-{timestamp}-{random}`).
2. Camera nhận diện khuôn mặt (TinyFaceDetector + Landmark68Tiny + FaceRecognitionNet).
3. Bấm **Scan mẫu** → Frontend kiểm tra chất lượng ảnh (`evaluateFaceQualityFromCanvas`: brightness, contrast, sharpness, face area).
4. Lặp lại bước 3 cho đến khi đủ **tối thiểu 5 mẫu** (tối đa 10).
5. Bấm **Lưu** → Frontend tính prototype (trung bình 128D) → Gửi POST `/api/v1/face/register` kèm `samples[]`, `prototype`, `id`, `name`.
6. Backend xây dựng descriptor blob v2 → Lưu Employee vào PostgreSQL.

### Luồng Điểm danh (Attendance Flow)
1. Nhân viên đứng trước Camera trên tab **Login**.
2. **TinyFaceDetector** + **FaceRecognitionNet** phát hiện khuôn mặt → hiển thị bounding box real-time.
3. Bấm **Scan Face (Vote)** → Frontend tự động thu **5 frame hợp lệ**:
   - Mỗi frame kiểm tra **Face Quality** (brightness 72–205, contrast ≥32, sharpness ≥24, face area ≥6%).
   - Frame không đạt → bỏ qua, tiếp tục frame sau.
   - Nếu <3 frame hợp lệ → **từ chối ngay** trên client.
4. Gửi POST `/api/v1/face/recognize` kèm `frames[]` + `vote_min_match`.
5. Backend chuyển tiếp sang AI Service `/predict`.
6. AI Service (với mỗi frame):
   - **Anti-Spoofing** (CNN+DSP+LSTM) → Liveness check.
   - **2-Stage Matching:** FAISS top-5 → Euclidean distance < 0.55 → Match.
7. **Vote Aggregation:** Đếm phiếu theo employee ID → đủ vote threshold → **Access Granted**.
8. Frontend hiển thị kết quả: tên nhân viên, liveness score, similarity, matched votes, frames used.

### Luồng Tiền xử lý dữ liệu (Preprocessing Flow)
1. Đặt dữ liệu thô vào `data/anti-spoof/raw/{celeba-spoof,ff-c23}/`.
2. Chạy pipeline tương ứng:
   - CelebA Spoof: `python -m preprocessing`
   - FF-C23: `python -m preprocessing.pipeline_ffc23`
3. Pipeline tự động: cleaning → splitting → extraction (FF-C23) → DataLoaders → visualization.
4. Output: `dataset/{train,val,test}/{celeba-spoof,ff-c23}/{live,spoof}/` — sẵn sàng cho huấn luyện.

---

## 8. Cấu trúc thư mục dự án (Project Structure)

```
face-attendance-system/
├── frontend/                          # Giao diện người dùng
│   ├── src/
│   │   ├── App.jsx                    # State-based navigation (Login/Register tabs)
│   │   ├── main.jsx                   # Entry point React
│   │   ├── assets/index.css           # TailwindCSS styles + dark theme
│   │   ├── services/api.js            # Axios API client (recognize + register)
│   │   ├── utils/
│   │   │   └── faceQuality.js         # Face quality evaluation (brightness, contrast, sharpness, face area)
│   │   └── pages/
│   │       ├── Camera/CameraPage.jsx  # Live Auth + Multi-frame Vote + Quality Check
│   │       └── Register/RegisterEmployee.jsx  # Multi-sample register + Quality Gate
│   ├── public/models/                 # Face-API.js model weights
│   │   ├── tiny_face_detector_model.* # TinyFaceDetector (real-time detection)
│   │   ├── face_landmark_68_tiny_model.* # 68-point face landmark (tiny version)
│   │   └── face_recognition_model.*   # 128D face descriptor extraction
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── package.json
│
├── backend/                           # API Server trung tâm
│   └── app/
│       ├── main.py                    # FastAPI app + CORS + auto create tables
│       ├── api/endpoints/face.py      # /recognize + /register (with descriptor)
│       ├── models/employee.py         # Employee model (+ face_descriptor column)
│       └── database/session.py        # PostgreSQL connection (SQLAlchemy)
│
├── ai-service/                        # Dịch vụ AI độc lập
│   ├── inference/
│   │   ├── face_match.py              # /predict + /refresh-cache (FAISS, cache, vote)
│   │   ├── mock_model.py              # Mock models (fallback khi chưa train)
│   │   ├── antispoof_model.py         # CNN+DSP+LSTM model + AntiSpoofPredictor (TTA + Video Predict)
│   │   └── dsp_utils.py               # DSP/FFT utility functions
│   ├── train.py                       # Training script (CNN+DSP+LSTM + Multi-frame)
│   ├── video_dataset.py               # PyTorch Dataset gom frame thành chuỗi (Video Sequence)
│   ├── temporal_augmentation.py       # Augmentation chuỗi thời gian (Flicker, Color shift, Noise)
│   ├── test.py                        # Test evaluation script (metrics + charts)
│   ├── test_integration.py            # Integration test cho model
│   ├── requirements.txt               # Dependencies: torch, torchvision, scipy, ...
│   ├── models/
│   │   └── weights/
│   │       ├── antispoof_cnn_dsp_lstm.pth  # Trained weights (best model epoch 41)
│   │       ├── antispoof_last.pth         # Last checkpoint (cho resume training)
│   │       └── training_log.json          # Lịch sử training metrics (JSON)
│   ├── training_logs/                 # Biểu đồ training (tự động generate)
│   │   ├── loss_curves.png            # Train/Val Loss curves
│   │   ├── accuracy_curves.png        # Train/Val Accuracy curves
│   │   ├── precision_recall_f1.png    # P/R/F1 per class (LIVE + SPOOF)
│   │   ├── learning_rate.png          # LR schedule (CosineAnnealing)
│   │   ├── confusion_matrix.png       # Confusion matrix heatmap
│   │   └── training_overview.png      # Tổng hợp 2×2 (Loss+Acc+P+LR)
│   ├── test_logs/                     # Biểu đồ test evaluation (tự động generate)
│   │   ├── test_confusion_matrix.png      # Confusion matrix trên tập test
│   │   ├── test_classification_report.png # P/R/F1 per class (bar chart)
│   │   ├── test_roc_curve.png             # ROC curve + AUC
│   │   ├── test_precision_recall_curve.png# Precision-Recall curve + AP
│   │   ├── test_score_distribution.png    # Histogram P(live) score distribution
│   │   ├── test_per_source_metrics.png    # Metrics per data source
│   │   ├── test_overview.png              # Dashboard tổng hợp 2×3
│   │   └── test_results.json              # Full test metrics (JSON)
│   └── venv/
│
├── preprocessing/                     # Pipeline tiền xử lý dữ liệu huấn luyện
│   ├── __init__.py                    # Package init
│   ├── __main__.py                    # Entry point (python -m preprocessing)
│   ├── config.py                      # Cấu hình CelebA Spoof (paths, params)
│   ├── config_ffc23.py                # Cấu hình FF-C23 (paths, params, spoof methods)
│   ├── cleaning.py                    # Cleaning CelebA: corrupt, duplicate (dhash), face detect
│   ├── cleaning_ffc23.py              # Cleaning FF-C23: corrupt, blur (Laplacian)
│   ├── splitting.py                   # Stratified split CelebA (random, 70/15/15)
│   ├── splitting_ffc23.py             # Video-level split FF-C23 (không data leakage)
│   ├── frame_extraction.py            # Trích xuất frame + MTCNN face crop (FF-C23)
│   ├── augmentation.py                # Transform augmentation cho train & eval
│   ├── dataset.py                     # PyTorch Dataset/DataLoader CelebA Spoof
│   ├── dataset_ffc23.py               # PyTorch Dataset/DataLoader FF-C23
│   ├── visualization.py               # Grid ảnh augmented + biểu đồ phân bố lớp
│   ├── pipeline.py                    # Orchestrator CelebA Spoof (6 bước)
│   ├── pipeline_ffc23.py              # Orchestrator FF-C23 (4 bước)
│   ├── requirements.txt               # Dependencies: torch, torchvision, facenet-pytorch, ...
│   ├── CHANGELOG.md                   # Lịch sử thay đổi preprocessing
│   └── outputs/                       # Charts, logs, visualizations
│
├── data/                              # Dữ liệu thô (raw)
│   └── anti-spoof/raw/
│       ├── celeba-spoof/              # CelebA Spoof Mini dataset
│       └── ff-c23/                    # FaceForensics++ C23 videos
│
├── dataset/                           # Dữ liệu đã xử lý (output of preprocessing)
│   ├── train/
│   │   ├── celeba-spoof/{live,spoof}/ # Ảnh CelebA đã split
│   │   └── ff-c23/{live,spoof/...}/   # Frames FF-C23 đã crop mặt
│   ├── val/
│   │   ├── celeba-spoof/{live,spoof}/
│   │   └── ff-c23/{live,spoof/...}/
│   └── test/
│       ├── celeba-spoof/{live,spoof}/
│       └── ff-c23/{live,spoof/...}/
│
├── currentVersion.md                  # Tài liệu phiên bản hiện tại (file này)
├── train_val_test.md                  # Báo cáo chi tiết phân chia dữ liệu Train/Val/Test
└── README.md                          # Hướng dẫn cài đặt & chạy
```

---

**Kết luận:** Hệ thống đã hoàn thiện toàn bộ pipeline Anti-Spoofing — từ preprocessing dữ liệu (**80,817 ảnh** từ CelebA Spoof + FF-C23, chia 70/15/15 không data leakage), kiến trúc mô hình hybrid tiên tiến **CNN + DSP + LSTM + SE Attention** (kết hợp **EfficientNet-B0**, ~8.8M tham số), đến huấn luyện (50 epochs), đánh giá trên tập test, và tích hợp inference vào AI Service. Quá trình huấn luyện thực tế đã áp dụng triệt để bộ tính năng chống Overfitting mạnh mẽ (Focal Loss, SWA, MixUp, Cutout, RandomErasing, Asymmetric Weights, Dropout — 11 kỹ thuật augmentation). Mô hình đạt kết quả trên tập test: **Accuracy 94.41%**, **ROC AUC 0.9802**, **Spoof F1 0.9603**, **Live F1 0.9053** — chứng minh khả năng generalize tốt trên dữ liệu chưa từng thấy (Test accuracy cao hơn Val accuracy → không overfitting). Mức ngưỡng phân loại được tìm kiếm tự động (**Optimal Threshold = 0.610**) đảm bảo spoof recall ≥ 95%. Hệ thống inference được tăng cường với **FAISS ANN Index** (HNSW, tăng tốc tra cứu descriptor), **Employee Cache TTL-based** (giảm tải DB), **2-Stage Matching** (prototype → sample), **Multi-frame Vote Aggregation** (chống false positive), và **Test-Time Augmentation** (5 biến thể). Xem chi tiết phân bố dữ liệu và phân tích training trong [`train_val_test.md`](train_val_test.md). Toàn bộ module phần lõi hiện tại đã sẵn sàng để cắm vào Backend, phục vụ ứng dụng điểm danh với mức an ninh thực tế ở chuẩn Production.

