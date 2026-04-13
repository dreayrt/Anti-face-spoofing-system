from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import json
import base64
import cv2
import os

# Import the mock models (fallback)
from mock_model import MockAntiSpoofModel, MockFaceRecognitionModel

app = FastAPI(title="AI Inference Service")

# ── Anti-Spoofing Model Initialization ───────────────────────────────────
# Try to load the real CNN+DSP+LSTM model; fallback to mock if no checkpoint
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "weights", "antispoof_cnn_dsp_lstm.pth"
)

spoof_model = None
USE_REAL_MODEL = False

try:
    from antispoof_model import AntiSpoofPredictor
    if os.path.isfile(CHECKPOINT_PATH):
        spoof_model = AntiSpoofPredictor(checkpoint_path=CHECKPOINT_PATH)
        USE_REAL_MODEL = True
        print("[AI Service] ✅ Loaded REAL Anti-Spoofing Model (CNN+DSP+LSTM)")
    else:
        print(f"[AI Service] ⚠️ Checkpoint not found: {CHECKPOINT_PATH}")
        print("[AI Service]    Falling back to MockAntiSpoofModel")
        spoof_model = MockAntiSpoofModel(weights_path="models/weights/antispoof_model.h5")
except ImportError as e:
    print(f"[AI Service] ⚠️ Could not import AntiSpoofPredictor: {e}")
    print("[AI Service]    Falling back to MockAntiSpoofModel")
    spoof_model = MockAntiSpoofModel(weights_path="models/weights/antispoof_model.h5")

recognition_model = MockFaceRecognitionModel(weights_path="models/weights/facenet_model.pt")

# Database connection config (same DB as the backend)
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "dbname": os.getenv("DB_NAME", "FaceDetect"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "123456"),
}

class InferenceRequest(BaseModel):
    image_base64: str
    box: dict = None
    descriptor: Optional[List[float]] = None  # 128D face descriptor from face-api.js


def decode_base64_to_image(base64_str: str):
    """Decode a base64 image string (with or without data URI prefix) to an OpenCV image."""
    if "," in base64_str:
        base64_data = base64_str.split(",")[1]
    else:
        base64_data = base64_str
    img_data = base64.b64decode(base64_data)
    np_arr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def get_registered_employees():
    """Fetch all registered employees (with face descriptors) from the PostgreSQL database."""
    import psycopg2
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, name, face_descriptor FROM employees")
        rows = cur.fetchall()
        cur.close()
        employees = []
        for r in rows:
            descriptor = None
            if r[2]:
                try:
                    descriptor = json.loads(r[2])
                except json.JSONDecodeError:
                    pass
            employees.append({"id": r[0], "name": r[1], "descriptor": descriptor})
        return employees
    finally:
        conn.close()


def euclidean_distance(desc1, desc2):
    """Compute Euclidean distance between two face descriptors."""
    a = np.array(desc1, dtype=np.float64)
    b = np.array(desc2, dtype=np.float64)
    return float(np.linalg.norm(a - b))


@app.post("/predict")
async def predict_face(request: InferenceRequest):
    """
    Main endpoint for the AI Worker.
    1. Decode Base64 image
    2. Run anti-spoofing check
    3. Match face descriptor against registered employees in DB
    """
    try:
        # Decode base64 to OpenCV image (numpy array)
        img = decode_base64_to_image(request.image_base64)

        if img is None:
            raise ValueError("Invalid image data")

        # 1. Image Preprocessing & Cropping
        face_crop = img 
        if request.box:
            h_img, w_img, _ = img.shape
            x = max(0, int(request.box.get('x', 0)))
            y = max(0, int(request.box.get('y', 0)))
            w = int(request.box.get('w', w_img))
            h = int(request.box.get('h', h_img))
            
            if w > 0 and h > 0:
                margin_x = int(w * 0.1)
                margin_y = int(h * 0.1)
                
                x1 = max(0, x - margin_x)
                y1 = max(0, y - margin_y)
                x2 = min(w_img, x + w + margin_x)
                y2 = min(h_img, y + h + margin_y)
                
                face_crop = img[y1:y2, x1:x2]
                
                if face_crop.size == 0:
                    face_crop = img

        # 2. Anti-Spoofing Check
        liveness_score = spoof_model.predict(face_crop)
        is_real = liveness_score > 0.8
        
        if not is_real:
             return {
                 "is_real": False, 
                 "matched": False, 
                 "liveness_score": liveness_score,
                 "user": None
             }

        # 3. Face Descriptor Matching — compare with registered employees
        if not request.descriptor:
            return {
                "is_real": True,
                "matched": False,
                "liveness_score": liveness_score,
                "user": None,
                "message": "No face descriptor provided by client"
            }

        employees = get_registered_employees()

        if not employees:
            return {
                "is_real": True,
                "matched": False,
                "liveness_score": liveness_score,
                "user": None
            }

        best_match = None
        smallest_distance = float('inf')

        for emp in employees:
            if not emp["descriptor"]:
                print(f"[FaceMatch] Skipping {emp['name']} — no descriptor stored")
                continue

            # Euclidean distance: lower = more similar
            distance = euclidean_distance(request.descriptor, emp["descriptor"])
            print(f"[FaceMatch] {emp['name']} (ID: {emp['id']}): distance = {distance:.4f}")

            if distance < smallest_distance:
                smallest_distance = distance
                best_match = emp

        # face-api.js threshold: distance < 0.45 = same person (strict)
        MATCH_THRESHOLD = 0.45

        if best_match is None or smallest_distance >= MATCH_THRESHOLD:
            print(f"[FaceMatch] ❌ No match found (best distance: {smallest_distance:.4f}, threshold: {MATCH_THRESHOLD})")
            return {
                "is_real": True,
                "matched": False,
                "liveness_score": liveness_score,
                "user": None
            }

        matched_user = {
            "id": best_match["id"],
            "name": best_match["name"],
        }

        print(f"[FaceMatch] ✅ Match: {matched_user['name']} (distance: {smallest_distance:.4f})")

        return {
            "is_real": True,
            "matched": True,
            "liveness_score": liveness_score,
            "user": matched_user,
            "similarity_score": round(1.0 - smallest_distance, 4)
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # The AI service runs on a different port (8001) to keep the web API (8000) free
    uvicorn.run(app, host="0.0.0.0", port=8001)

