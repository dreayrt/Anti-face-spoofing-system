# Hệ Thống Điểm Danh Bằng Khuôn Mặt & Chống Giả Mạo (Face Attendance & Anti-Spoofing System)

Dự án này là một kiến trúc vi dịch vụ (microservice) gồm 3 phần: Frontend React, Backend FastAPI và Dịch vụ AI bằng Python.

## 📂 Tổng quan cấu trúc dự án

- **`frontend/`**: Ứng dụng React sử dụng Vite và TailwindCSS. Chạy trên cổng `5173`.
- **`backend/`**: Máy chủ chính FastAPI xử lý logic nghiệp vụ, kết nối cơ sở dữ liệu và định tuyến API. Chạy trên cổng `8000`.
- **`ai-service/`**: Worker FastAPI chạy mô hình Deep Learning **CNN + DSP + LSTM** để chống giả mạo khuôn mặt (anti-spoofing) và khớp khuôn mặt (face matching). Chạy trên cổng `8001`.
- **`preprocessing/`**: Pipeline tiền xử lý dữ liệu huấn luyện từ CelebA Spoof và FaceForensics++ (FF-C23).

---

## 🚀 Hướng dấn chạy dự án trên máy Cục Bộ (Local)

Bạn sẽ cần chạy ba cửa sổ dòng lệnh (terminal/tab) riêng biệt, tương ứng với mỗi service.

### 1. Khởi chạy AI Service (Terminal 1)

Service này load mô hình Anti-Spoofing **CNN + DSP + LSTM** (nếu đã train) hoặc fallback về Mock Model.

```bash
cd face-attendance-system/ai-service

# Tạo và kích hoạt môi trường ảo (Khuyến khích)
python -m venv venv
# Trên Windows:
venv\Scripts\activate
# Trên Mac/Linux:
# source venv/bin/activate

# Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt

# (Tùy chọn) Chạy integration test để kiểm tra model
python test_integration.py

# Chạy server xử lý AI (inference)
python inference/face_match.py
```
*Dịch vụ AI hiện sẽ chạy tại địa chỉ `http://localhost:8001`*

> **Lưu ý:** Khi khởi động, AI Service sẽ tự động tìm checkpoint tại `models/weights/antispoof_cnn_dsp_lstm.pth`. Nếu chưa train → dùng MockModel. Nếu đã train → dùng model thật.

---

### 2. Khởi chạy Backend chính (Terminal 2)

Service này kết nối frontend với cơ sở dữ liệu và chuyển tiếp các tác vụ AI nặng sang AI Service.

```bash
cd face-attendance-system/backend

# Tạo và kích hoạt môi trường ảo (Khuyến khích)
python -m venv venv
# Trên Windows:
venv\Scripts\activate
# Trên Mac/Linux:
# source venv/bin/activate

# Cài đặt các thư viện phụ thuộc
pip install fastapi uvicorn sqlalchemy psycopg2-binary pydantic python-multipart httpx

# Chạy server backend
uvicorn app.main:app --reload --port 8000
```
*API Backend hiện sẽ chạy tại địa chỉ `http://localhost:8000`*
*(Bạn có thể xem tài liệu API Swagger tương tác tại `http://localhost:8000/docs`)*

---

### 3. Khởi chạy React Frontend (Terminal 3)

Giao diện để người dùng xem bảng điều khiển (dashboard) và tương tác với kiosk camera.

```bash
cd face-attendance-system/frontend

# Cài đặt các thư viện Node.js
npm install

# Chạy server phát triển Vite
npm run dev
```
*Frontend hiện sẽ chạy tại địa chỉ `http://localhost:5173` (hoặc tương tự, hãy kiểm tra output trong terminal).*

---

## 🤖 Huấn luyện mô hình Anti-Spoofing (Training)

Sau khi đã chạy preprocessing pipeline để chuẩn bị dữ liệu, bạn có thể huấn luyện mô hình CNN + DSP + LSTM:

```bash
cd face-attendance-system/ai-service

# Kích hoạt môi trường ảo
venv\Scripts\activate

# Train với cấu hình mặc định (MobileNetV2, 30 epochs, lr=0.0001)
python train.py

# Train với backbone khác
python train.py --backbone resnet50
python train.py --backbone efficientnet_b0

# Tùy chỉnh hyperparameters
python train.py --epochs 50 --lr 0.0001 --batch-size 16

# Resume training từ checkpoint trước
python train.py --resume
```

**Output sau khi train:**
- `models/weights/antispoof_cnn_dsp_lstm.pth` — Best model checkpoint
- `models/weights/training_log.json` — Lịch sử training metrics

> Sau khi train xong, chỉ cần khởi động lại AI Service — model thật sẽ được tự động load.

---

## 📊 Tiền xử lý dữ liệu (Preprocessing)

```bash
cd face-attendance-system

# Kích hoạt môi trường ảo preprocessing
cd preprocessing
pip install -r requirements.txt
cd ..

# Chạy pipeline CelebA Spoof
python -m preprocessing

# Chạy pipeline FaceForensics++ (FF-C23)
python -m preprocessing.pipeline_ffc23
```

Output: `dataset/{train,val,test}/{celeba-spoof,ff-c23}/{live,spoof}/`

---

## 🧪 Kiểm tra hoạt động của hệ thống

1. Mở trình duyệt và truy cập vào URL của Frontend (`http://localhost:5173`).
2. Nhấn vào **"Camera / Kiosk"** trên thanh điều hướng.
3. Nhấn **"Start Camera"** (Bắt đầu máy ảnh) và cấp quyền cho trình duyệt sử dụng webcam.
4. Nhấn **"Verify Attendance"** (Xác thực điểm danh).

**Những gì diễn ra bên dưới hệ thống:**
1. Ứng dụng React chụp một khung hình (frame) từ webcam và chuyển nó thành chuỗi base64.
2. Nó gửi một yêu cầu HTTP POST đến `http://localhost:8000/api/v1/face/recognize`.
3. Backend FastAPI nhận yêu cầu và dùng `httpx` để chuyển tiếp (forward) chuỗi tới `http://localhost:8001/predict`.
4. AI Service chạy mô hình **CNN + DSP + LSTM** (hoặc MockModel nếu chưa train) để kiểm tra liveness.
5. Nếu khuôn mặt thật → so khớp Face Descriptor 128D với CSDL bằng Euclidean Distance.
6. Kết quả đi ngược lại về phía giao diện React (UI) để hiển thị thông báo thành công hay thất bại.
