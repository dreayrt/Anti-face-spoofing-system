# Nguồn Dữ Liệu Thô Ban Đầu (Raw Data Sources)

Hệ thống Face Attendance & Anti-Spoofing sử dụng 3 bộ dữ liệu thô ban đầu để huấn luyện và đánh giá mô hình chống giả mạo (Anti-Spoofing). Dưới đây là thông tin chi tiết và ý nghĩa của từng bộ dữ liệu:

## 1. CelebA-Spoof
- **Ý nghĩa:** Đây là bộ dữ liệu ảnh tĩnh quy mô lớn được xây dựng dựa trên bộ dữ liệu khuôn mặt nổi tiếng CelebA. Nó cung cấp các mẫu tấn công giả mạo (spoofing) đa dạng, đặc biệt tập trung vào các dạng tấn công bằng ảnh in (print attack) và phát lại qua màn hình (replay attack) dưới nhiều điều kiện ánh sáng và môi trường khác nhau. Giúp mô hình nhận diện tốt các trường hợp dùng ảnh thật hoặc điện thoại giơ trước camera.
- **Loại dữ liệu:** Ảnh tĩnh (Images).
- **Số lượng dòng/files gốc:** **13,881** files.

## 2. FaceForensics++ (FF-C23)
- **Ý nghĩa:** Bộ dữ liệu tiên phong và lớn nhất về phát hiện video giả mạo sâu (Deepfake). Phiên bản C23 (mức nén H.264 với tỷ lệ CRF 23) mô phỏng video nén chất lượng cao trên internet. Bộ dữ liệu này tập trung vào các kỹ thuật thao túng khuôn mặt bằng AI bao gồm: *Deepfakes*, *FaceSwap*, *Face2Face*, *NeuralTextures*, và *FaceShifter*. Đóng vai trò cực kỳ quan trọng giúp nhánh xử lý tín hiệu số (DSP) của mô hình học được các "dấu vết" (artifacts) và viền giả mạo sinh ra từ các mô hình AI tạo sinh.
- **Loại dữ liệu:** Video (sau quá trình tiền xử lý sẽ được trích xuất thành các khung hình tĩnh - frames).
- **Số lượng dòng/files gốc:** **7,010** files (video thô gốc). *Lưu ý: Qua bước tiền xử lý, hệ thống đã trích xuất ra thành 66,936 frames.*

## 3. SiW (Spoof in the Wild)
- **Ý nghĩa:** Bộ dữ liệu kiểm tra giả mạo khuôn mặt trong môi trường "hoang dã" (thực tế). Nó thu thập dữ liệu với sự đa dạng cao về góc mặt, ánh sáng, nền, khoảng cách camera và biểu cảm khuôn mặt. SiW giúp đánh giá độ bền vững (robustness) và khả năng tổng quát hóa của mô hình đối với các cuộc tấn công Presentation Attacks (PA) thực tế. Điểm đặc biệt của tập này là dữ liệu được quản lý theo đối tượng (Subject ID), ngăn chặn tuyệt đối hiện tượng rò rỉ dữ liệu (data leakage) khi train/val/test.
- **Loại dữ liệu:** Video/Ảnh tập trung vào các màn tấn công trực tiếp.
- **Số lượng dòng/files gốc:** **7,586** files.

---

### Tổng Kết Số Lượng File Trong Thư Mục Raw:
- **CelebA-Spoof:** 13,881
- **FF-C23:** 7,010
- **SiW:** 7,586
- **Tổng cộng:** **28,477 files** thô ban đầu.
