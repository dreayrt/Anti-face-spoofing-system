# Phân tích Chuyên sâu: Kỹ thuật Tiền xử lý Dữ liệu & Data Augmentation trong Hệ thống Anti-Spoofing

Tài liệu này giải thích chi tiết các kỹ thuật tiền xử lý dữ liệu (Data Preprocessing), làm sạch (Cleaning) và gia tăng dữ liệu (Data Augmentation) được áp dụng trong hệ thống Face Anti-Spoofing.

Hệ thống chia quy trình xử lý thành 2 giai đoạn chính:
1. **Tiền xử lý Offline (Preprocessing Pipeline)**: Chuẩn bị, làm sạch và chia tách dữ liệu trước khi train.
2. **Gia tăng dữ liệu On-the-fly (Data Augmentation & Sampling)**: Biến đổi dữ liệu động trong quá trình huấn luyện mô hình.

---

## PHẦN 1: TIỀN XỬ LÝ OFFLINE (OFFLINE PREPROCESSING)
Giai đoạn này được thực hiện bởi các script trong thư mục `preprocessing/` để tạo ra bộ dữ liệu sạch trong thư mục `dataset/`.

### 1. Trích xuất & Cắt khuôn mặt (Face Extraction & Cropping)
- **Áp dụng tại:** Pipeline xử lý video (FaceForensics++ / FF-C23).
- **Kỹ thuật:** Dùng **MTCNN** (Multi-task Cascaded Convolutional Networks) hoặc **Haar Cascade**. Hệ thống lấy từng khung hình (frame) từ video, tìm vị trí khuôn mặt và cắt ra kèm theo một khoảng lề (margin/padding) nhất định.
- **Tác dụng:** 
  - Loại bỏ bối cảnh nền (background) không cần thiết, ép mô hình Deep Learning chỉ tập trung học các đặc trưng giả mạo trên khuôn mặt và vùng lân cận.
  - Việc lấy thêm margin giúp model thấy được viền khuôn mặt hoặc thiết bị cầm tay (nếu có).

### 2. Trích xuất khung hình giảm dư thừa (Frame Sampling)
- **Áp dụng tại:** Pipeline FF-C23 (Video Dataset).
- **Kỹ thuật:** Cứ mỗi 10 frames mới trích xuất 1 frame (Frame Sample Rate = 10), giới hạn tối đa 30 frames cho mỗi video.
- **Tác dụng:** Trong 1 video, các frame liên tiếp thường rất giống nhau (temporal redundancy). Việc lấy mẫu thưa giúp giảm dung lượng lưu trữ, tăng tốc độ xử lý mà không làm mất đi tính đa dạng của dữ liệu.

### 3. Lọc ảnh mờ (Laplacian Blur Detection)
- **Áp dụng tại:** `preprocessing/cleaning_ffc23.py`
- **Kỹ thuật:** Chuyển ảnh sang thang độ xám (grayscale) và áp dụng bộ lọc đạo hàm bậc hai Laplacian. Sau đó tính phương sai (Variance) của kết quả. Nếu Variance < 50.0, ảnh bị coi là quá mờ và bị loại bỏ.
- **Tác dụng:** Đảm bảo dữ liệu huấn luyện có độ sắc nét nhất định. Ảnh quá mờ sẽ làm mất các chi tiết kết cấu (texture) hoặc nhiễu tần số (frequency artifacts) quan trọng dùng để phân biệt thật/giả.

### 4. Loại bỏ ảnh trùng lặp (Perceptual Hashing - dhash)
- **Áp dụng tại:** Pipeline CelebA Spoof (Ảnh tĩnh).
- **Kỹ thuật:** Sử dụng thuật toán `dhash` (Difference Hash) để băm ảnh thành 1 chuỗi bit dựa trên chênh lệch độ sáng. Tính khoảng cách Hamming giữa các hash, nếu Hamming distance $\le 5$, hai ảnh bị coi là trùng lặp.
- **Tác dụng:** Ngăn chặn việc mô hình học đi học lại một mẫu giống hệt nhau, chống Overfitting và tránh việc ảnh trùng lặp xuất hiện ở cả tập Train và tập Test (gây rò rỉ dữ liệu).

### 5. Xóa ảnh hỏng (Corrupt Data Removal)
- **Áp dụng tại:** Tất cả các pipeline.
- **Kỹ thuật:** Sử dụng hàm `PIL.Image.verify()` và thử load ảnh vào bộ nhớ. Nếu ném ra Exception, file bị xóa.
- **Tác dụng:** Ngăn chặn DataLoader bị crash giữa chừng trong quá trình training kéo dài nhiều giờ.

### 6. Video-level Splitting (Chia dữ liệu mức độ Video)
- **Áp dụng tại:** `preprocessing/splitting_ffc23.py`
- **Kỹ thuật:** Thay vì gộp tất cả frames lại rồi chia ngẫu nhiên, hệ thống **chia theo ID của Video gốc**. Tất cả frames thuộc Video A sẽ hoàn toàn nằm ở tập Train hoặc hoàn toàn ở tập Test.
- **Tác dụng:** Tránh Data Leakage (Rò rỉ dữ liệu). Nếu frame 1 của Video A nằm ở Train, và frame 2 nằm ở Test, model sẽ gian lận vì hai frame này gần như giống hệt nhau. Tính chính xác khi test sẽ bị thổi phồng một cách không thực tế.

---

## PHẦN 2: GIA TĂNG DỮ LIỆU ON-THE-FLY (ON-THE-FLY AUGMENTATION)

**Mục đích cốt lõi:** 
1. **Chống Overfitting (học vẹt):** Ép mô hình học đặc trưng bản chất thay vì ghi nhớ ảnh.
2. **Mô phỏng thực tế:** Giả lập các điều kiện môi trường nhiễu (ánh sáng, góc camera, mờ).
3. **Tiết kiệm chi phí:** Tự động tạo ra đa dạng biến thể mà không cần đi thu thập thêm dữ liệu mới.
4. **Tránh phụ thuộc chi tiết cục bộ:** Ép mô hình nhìn tổng thể, không được lười biếng dựa dẫm vào 1 điểm quen thuộc (nhờ các kỹ thuật che ảnh).

*Được áp dụng ngẫu nhiên trong mỗi batch lúc training (nằm trong `train.py`, `video_dataset.py`, `temporal_augmentation.py`).*

### 1. Spatial Augmentations (Gia tăng Không gian cơ bản)
- **Kỹ thuật:** `RandomHorizontalFlip` (lật ngang), `RandomRotation` (xoay $\pm 15^\circ$), `ColorJitter` (thay đổi độ sáng, độ tương phản), `GaussianBlur` (làm mờ ngẫu nhiên), `RandomPerspective`, `RandomAffine`.
- **Tác dụng:** Giả lập các điều kiện môi trường thực tế (góc mặt khác nhau, ánh sáng chói/tối, camera mờ). Ép mô hình phải học đặc trưng cốt lõi thay vì ghi nhớ (memorize) ảnh gốc.

### 2. Cutout & RandomErasing
- **Kỹ thuật:** 
  - **Cutout**: Xóa (tô đen) ngẫu nhiên các vùng hình chữ nhật nhỏ trên ảnh.
  - **RandomErasing**: Xóa các vùng trên tensor (sau khi chuẩn hóa) bằng các giá trị pixel ngẫu nhiên.
- **Tác dụng:** Buộc mạng CNN không được phụ thuộc vào bất kỳ một chi tiết cụ thể nào (ví dụ: chỉ nhìn vào mắt để đoán). Nếu vùng mắt bị che, model phải học cách dùng mũi, miệng hoặc kết cấu da để nhận diện. Cực kỳ hiệu quả chống Overfitting.

### 3. MixUp Augmentation
- **Kỹ thuật:** Lấy 2 ảnh ngẫu nhiên (ví dụ ảnh A là Live, ảnh B là Spoof) và trộn lẫn (blend) vào nhau theo hệ số alpha. Nhãn cũng được trộn mềm (ví dụ: 0.7 Live + 0.3 Spoof).
- **Tác dụng:** 
  - Làm mịn ranh giới quyết định (decision boundary) của mô hình.
  - Giảm sự tự tin thái quá (overconfidence) của mô hình, giúp model tổng quát hóa tốt hơn trên các kiểu tấn công spoofing chưa từng gặp.

### 4. Temporal Augmentation (Gia tăng Thời gian)
- **Áp dụng tại:** `ai-service/temporal_augmentation.py` (Áp dụng cho chuỗi Video Sequence).
- **Kỹ thuật:** Chỉ áp dụng cho các video giả mạo (Spoof).
  - **Flicker**: Thêm mức độ sáng/tối khác nhau cho từng frame trong 1 chuỗi.
  - **Color Shift**: Lệch màu nhẹ giữa các frame liên tiếp.
  - **Per-frame Noise**: Áp dụng ma trận nhiễu Gauss khác biệt cho từng frame.
- **Tác dụng:** Trong thực tế, khi cầm điện thoại (chứa video giả mạo) giơ trước camera, tốc độ làm tươi (refresh rate) của 2 màn hình sẽ lệch nhau tạo ra hiện tượng nhấp nháy, nhiễu hạt không nhất quán theo thời gian. Kỹ thuật này cố tình tạo ra các hiệu ứng đó để dạy cho lớp mạng **LSTM** nhận biết: *Sự thiếu nhất quán theo thời gian = Fake (Spoof)*.

---

## PHẦN 3: XỬ LÝ MẤT CÂN BẰNG DỮ LIỆU (IMBALANCE HANDLING)
Do dữ liệu ảnh giả (Spoof) thường nhiều hơn ảnh thật (Live).

### 1. Weighted Random Sampler
- **Kỹ thuật:** Tính toán tần suất xuất hiện của lớp Live và Spoof. Cấp cho lớp Live trọng số lấy mẫu cao hơn. Khi DataLoader bốc ngẫu nhiên tạo batch, nó sẽ ưu tiên bốc lớp Live nhiều hơn.
- **Tác dụng:** Đảm bảo mỗi batch huấn luyện đều có tỷ lệ Live:Spoof xấp xỉ 50:50, giúp mô hình không bị "thiên lệch" (bias) chẩn đoán mọi thứ đều là Spoof chỉ vì lớp Spoof quá đông.

### 2. Focal Loss & Asymmetric Weights
- **Kỹ thuật:** 
  - **Focal Loss**: Giảm trọng số của các mẫu "dễ" (model đã đoán đúng với độ tự tin cao) và dồn sự tập trung vào các mẫu "khó" (model đang đoán sai).
  - **Asymmetric Weights**: Nhân hình phạt (loss penalty) của lớp Spoof lên gấp 2.5 - 3 lần.
- **Tác dụng:** Tránh hệ quả False Positive (Nhận nhầm kẻ gian là người thật). Hệ thống thà từ chối nhầm một người thật (bắt họ quét lại), còn hơn là cho phép một khuôn mặt giả mạo vượt qua hệ thống. Trọng số bất đối xứng giúp điều hướng mô hình thiên về tính an toàn.

---

**TỔNG KẾT:** Sự kết hợp của toàn bộ các bước Preprocessing, Cleaning và Augmentation phía trên là lý do chính giúp kiến trúc CNN+DSP+LSTM của bạn đạt được **Test Accuracy > 94%** mà hoàn toàn không bị Overfitting (Test Acc > Val Acc). Model đã được ép phải học bản chất thực sự của sự giả mạo (không gian, tần số, và thời gian) thay vì chỉ học thuộc lòng dữ liệu.
