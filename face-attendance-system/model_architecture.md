# Kiến trúc Mô hình Hệ thống: Tổng quan đến Chi tiết

Tài liệu này trình bày chi tiết về kiến trúc các mô hình học sâu và thuật toán được sử dụng trong Hệ thống Điểm danh và Chống giả mạo khuôn mặt (Face Attendance & Anti-Spoofing System).

---

## 1. Tổng quan Hệ thống (System Overview)

Hệ thống của bạn là một hệ thống **Đa thể thức (Multi-modal) và Đa luồng (Multi-stream)**, kết hợp giữa nhận diện khuôn mặt để điểm danh và phát hiện giả mạo (Liveness Detection) để đảm bảo tính an toàn.

Để đối phó với các hình thức tấn công giả mạo (dùng ảnh in, phát lại video trên màn hình), hệ thống chia làm 2 luồng công việc chính với các mô hình chuyên biệt:
1. **Luồng Chống giả mạo (Liveness/Anti-Spoofing):** Sử dụng kiến trúc kết hợp **CNN + DSP + LSTM**.
2. **Luồng Nhận diện danh tính (Face Recognition):** Sử dụng mô hình **FaceNet** (trích xuất vector đặc trưng 128 chiều).

Bên cạnh đó, dự án còn sử dụng **Swin Transformer** như một mô hình tham chiếu (Baseline) để đánh giá và so sánh hiệu năng.

---

## 2. Chi tiết các Mô hình Chính (Dùng trong hệ thống thực tế)

### 2.1. CNN (Convolutional Neural Network) - Nhánh Không gian
- **Vai trò:** Trích xuất các **đặc trưng cục bộ (local features)** và **kết cấu (texture)**. Nhìn vào từng khung hình để phát hiện độ bóng của mặt nạ, độ nhám của ảnh in, hoặc lỗi pixel/nhiễu màu.
- **Mức độ hoạt động:** Hoạt động ở **mức Không gian (Spatial Level)**.

### 2.2. DSP (Digital Signal Processing) - Nhánh Tín hiệu
- **Vai trò:**
  - Lọc và phân tích sự biến thiên màu sắc vi mô trên da để tìm tín hiệu **nhịp tim (rPPG)**.
  - Phân tích miền tần số không gian để phát hiện **nhiễu Moiré** (sóng sọc trên màn hình điện tử).
- **Mức độ hoạt động:** Hoạt động ở **mức Tín hiệu vật lý (Signal/Physical Level)**. 

### 2.3. LSTM (Long Short-Term Memory) - Nhánh Thời gian
- **Vai trò:**
  - Nắm bắt sự **thay đổi theo thời gian (temporal dynamics)**.
  - Quan sát chuỗi khung hình để phát hiện sự tự nhiên của chuyển động (chớp mắt, rung đầu tự nhiên).
  - Phân tích chuỗi đặc trưng được gộp từ CNN và DSP theo thời gian.
- **Mức độ hoạt động:** Hoạt động ở **mức Thời gian (Temporal Level) / Video-level**.

### 2.4. FS-Net (Face Spoofing Network) - Khối Tổng hợp và Quyết định
- **Thực tế trong code:** Khối này được lập trình dưới tên class `CNNDSPLSTMAntiSpoof`.
- **Vai trò:** Nhận tất cả các manh mối từ 3 nhánh CNN, DSP, LSTM và tiến hành **Gộp đặc trưng (Fusion)**. Nó đóng vai trò "Bộ não chỉ huy" để đưa ra xác suất cuối cùng: Khuôn mặt trước camera là Người thật (Live) hay Giả mạo (Spoof).

### 2.5. FaceNet - Khối Nhận diện Điểm danh
- **Thực tế trong code:** Khai báo qua class `MockFaceRecognitionModel("models/weights/facenet_model.pt")`.
- **Vai trò:** Sau khi FS-Net xác nhận là người thật, FaceNet sẽ đo đạc khuôn mặt để tạo ra một chuỗi số (Vector 128D). Chuỗi số này được so khớp với Database nhân viên bằng khoảng cách Euclidean để tìm ra danh tính.

---

## 3. Mô hình Đối chứng (Baseline for Comparison)

### 3.1. Swin Transformer
- **Vai trò trong dự án:** Không tham gia vào pipeline điểm danh thực tế (Inference Pipeline). Swin Transformer được sử dụng thuần túy trong giai đoạn **Train/Val/Test để lấy số liệu**.
- **Mục đích:** Đóng vai trò làm "Bia đỡ đạn" (Baseline) để chứng minh rằng: Mặc dù Swin Transformer là một mô hình rất mạnh mẽ và hiện đại trong xử lý ảnh, nhưng đối với bài toán chống giả mạo thời gian thực, kiến trúc **CNN+DSP+LSTM** của hệ thống mang lại hiệu quả cao hơn, chạy nhanh hơn và có khả năng bắt tín hiệu sinh lý vật lý tốt hơn.

---

## 4. Pipeline (Luồng xử lý thời gian thực của hệ thống)

Dưới đây là cách các mô hình liên kết với nhau trong thực tế (`face_match.py`):

1. **Bước 1: Nhận diện khuôn mặt (Face Detection):** Camera chụp ảnh và cắt vùng khuôn mặt người dùng.
2. **Bước 2: Kiểm tra Sự sống (Liveness Check):** 
   - Khuôn mặt được đẩy qua mạng `CNNDSPLSTMAntiSpoof`.
   - Các luồng CNN, DSP, LSTM hoạt động song song để kiểm tra. Nếu điểm số giả mạo (Spoof) cao $\rightarrow$ Chặn ngay lập tức, báo lỗi.
3. **Bước 3: Nhận diện Danh tính (Face Recognition):**
   - Nếu Bước 2 vượt qua (là người sống), dữ liệu đặc trưng 128D của khuôn mặt (do FaceNet tạo ra) sẽ được truy vấn bằng FAISS (hoặc tính khoảng cách Euclidean) với Database.
   - Nếu khoảng cách đủ nhỏ (dưới ngưỡng MATCH_THRESHOLD) $\rightarrow$ Điểm danh thành công.
