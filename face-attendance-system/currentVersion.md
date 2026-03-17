# Phiên bản hiện tại (Current Version) - Face Attendance & Anti-Spoofing System

Tài liệu này tổng hợp lại những tính năng, thành phần cốt lõi và cấu trúc mà hệ thống điểm danh bằng khuôn mặt (Face Attendance System) đã hoàn thiện cho đến thời điểm hiện tại.

Về cơ bản, hệ thống đã được thiết lập xong toàn bộ **khung kiến trúc (scaffolding)** với mô hình 3 dịch vụ (Monorepo Microservices) bao gồm Frontend, Backend và AI Service, có khả năng giao tiếp được với nhau từ đầu đến cuối (End-to-End flow).

---

## 1. Kiến trúc tổng thể (Architecture Scaffolding)
- Đã thiết lập xong cấu trúc thư mục rõ ràng chuẩn mô hình dự án lớn (Monorepo), phân tách rõ các miền: `frontend/`, `backend/`, `ai-service/`, `database/`.
- Quản lý dependencies (thư viện) độc lập cho từng dịch vụ (`package.json` cho frontend và `requirements.txt` cho backend/AI).

## 2. Giao diện người dùng - Frontend (React + Vite + TailwindCSS)
- **Cấu hình gốc:** Đã hoàn thiện toàn bộ công cụ, thư viện thiết yếu (`vite.config.js`, `postcss`, `tailwind`).
- **Dashboard (Bảng điều khiển):** Đã có trang thống kê hiển thị tổng quan số lượng nhân viên Điểm danh thành công, Vắng mặt và Tổng số (hiện đang dùng dữ liệu mô phỏng - mock data).
- **Camera Kiosk (Trạm quét khuôn mặt):**
  - Đã tích hợp thành công Web API để mở và kết nối trực tiếp với Webcam của máy tính.
  - Có khả năng chụp lại khung hình (frame) ngay tại thời điểm gọi điểm danh và chuyển đổi thành định dạng base64.
  - **Nhận diện khuôn mặt phía Client (Client-side Face Detection):** Đã tích hợp thư viện `@vladmandic/face-api` với model `TinyFaceDetector` để nhận diện khuôn mặt trực tiếp trên trình duyệt trước khi gửi ảnh về server. Hệ thống hiển thị bounding box xanh khi phát hiện 1 khuôn mặt, cảnh báo vàng khi không có mặt, và cảnh báo đỏ khi có nhiều hơn 1 mặt.
  - Nút "Scan Face" chỉ cho phép bấm khi detect đúng 1 khuôn mặt — kèm gửi tọa độ bounding box về Backend.
- **Tích hợp API:** Đã cấu hình Axios client (`services/api.js`) chuyên chịu trách nhiệm gửi hình ảnh thu thập được xuống Backend FastAPI.

## 3. Máy chủ xử lý trung tâm - Backend (FastAPI)
- **Khởi tạo Server:** Đã dựng xong khung chạy FastAPI kết hợp cấu hình **CORS** cho phép Frontend React (cổng 5173) có thể gọi API.
- **Kết nối CSDL (Database Session):** Đã thiết lập kết nối (SQLAlchemy Engine, Session) để chuẩn bị tương tác với PostgreSQL.
- **API Endpoints (`/api/v1/face/recognize`):**
  - Đã xây dựng logic nhận dữ liệu ảnh dạng base64 từ Frontend.
  - Hoạt động như một **Proxy/Gateway**, sử dụng thư viện `httpx` để chuyển tiếp ảnh bất đồng bộ (async) sang AI Service.
  - Phân tích và xử lý kết quả xác thực trước khi gửi thông báo (JSON) cuối cùng cho màn hình Kiosk.

## 4. Dịch vụ Trí tuệ nhân tạo - AI Service (Python)
- Dịch vụ chạy hoàn toàn độc lập với một cổng riêng (port 8001), chuyên biệt để tải (load) các model Deep Learning nặng.
- **API Xử lý lõi (`/predict`):** Có khả năng giải mã chuỗi base64 thành mảng hình ảnh (NumPy/OpenCV format) để phân tích.
- **Mô hình AI Giả lập (Mock Models):**
  - Đã triển khai khung class `MockAntiSpoofModel` để chấm điểm thật/giả (Liveness Score) chống giả mạo bằng ảnh in, video.
  - Đã triển khai khung class `MockFaceRecognitionModel` giả lập việc trích xuất vector khuôn mặt (embedding) và so sánh, định danh (Match) nhân viên.
  - Tích hợp thành công luồng mô phỏng: Nếu "giả mạo", hệ thống sẽ từ chối truy cập ngay lập tức, nếu "thật", hệ thống tiến hành đối chiếu khuôn mặt trong kho dữ liệu nhân viên.

---

## 5. Luồng hoạt động xuyên suốt (End-to-End Flow) đã hoạt động
1. Người dùng đứng trước Camera trên Web (Frontend).
2. Hình ảnh khuôn mặt của người dùng được Frontend chụp, đóng gói dưới dạng Base64 và Submit xuống Backend FastAPI.
3. Backend FastAPI kiểm tra gói tin và ngay lập tức gửi ảnh Base64 đó qua AI Service để xử lý chuyên sâu.
4. AI service giả lập chạy 2 mạng Neural Net (Anti-Spoofing và Face Matching), nếu tỷ lệ (score) qua ngưỡng an toàn, báo kết quả hợp lệ về cho Backend.
5. Frontend nhận kết quả cuối cùng từ Backend và hiển thị giao diện "Điểm danh thành công" tích xanh cho người dùng, kèm theo Tên nhân viên và dữ liệu độ tin cậy (Liveness Score).

---

**Kết luận:** Hệ thống hiện tại giống như một ngôi nhà đã xây xong khung thép, cấu trúc liên lạc điện nước. Các dịch vụ đã có thể "nói chuyện" được với nhau. Bước tiếp theo hoàn toàn là việc "lấp đầy": Lắp model AI thực tế vào file mock, và nối các hàm lưu/đọc dữ liệu log điểm danh vào database PostgreSQL.
