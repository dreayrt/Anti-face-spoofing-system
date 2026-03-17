from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import base64
import cv2

# Import the mock models
from mock_model import MockAntiSpoofModel, MockFaceRecognitionModel

app = FastAPI(title="AI Inference Service")

# Initialize models "in memory"
spoof_model = MockAntiSpoofModel(weights_path="models/weights/antispoof_model.h5")
recognition_model = MockFaceRecognitionModel(weights_path="models/weights/facenet_model.pt")

class InferenceRequest(BaseModel):
    image_base64: str
    box: dict = None

# Mock known employee DB
KNOWN_USERS = {
    "emp_101": {
        "id": "emp_101",
        "name": "Alice Developer",
        "embedding": np.random.rand(128).astype(np.float32) # Mock stored DB
    }
}

@app.post("/predict")
async def predict_face(request: InferenceRequest):
    """
    Main endpoint for the AI Worker.
    1. Decode Base64 image
    2. Run anti-spoofing check
    3. Run face recognition to identify user
    """
    try:
        # Decode base64 to OpenCV image (numpy array)
        if "," in request.image_base64:
            base64_data = request.image_base64.split(",")[1]
        else:
            base64_data = request.image_base64
            
        img_data = base64.b64decode(base64_data)
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Invalid image data")

        # 1. Image Preprocessing & Cropping
        # If the frontend sent bounding box coordinates, we use them to crop the face
        face_crop = img 
        if request.box:
            # Safely get coordinates and ensure they are integers within image bounds
            h_img, w_img, _ = img.shape
            x = max(0, int(request.box.get('x', 0)))
            y = max(0, int(request.box.get('y', 0)))
            w = int(request.box.get('w', w_img))
            h = int(request.box.get('h', h_img))
            
            # Crop the image array
            if w > 0 and h > 0:
                # Add a small margin (padding) around the face for better inference
                margin_x = int(w * 0.1)
                margin_y = int(h * 0.1)
                
                x1 = max(0, x - margin_x)
                y1 = max(0, y - margin_y)
                x2 = min(w_img, x + w + margin_x)
                y2 = min(h_img, y + h + margin_y)
                
                face_crop = img[y1:y2, x1:x2]
                
                # Check if crop is valid
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

        # 2. Face Recognition Match
        current_embedding = recognition_model.get_embedding(face_crop)
        
        # Searching through the database of known users
        # Mocking a match for our test user
        best_match = None
        highest_sim = 0
        
        # We will mock a success for demonstration. 
        # In reality, loop through KNOWN_USERS and use recognition_model.match
        
        # Mock returning a specific user
        matched_user = {
            "id": "emp_101",
            "name": "Jane Doe",
            "department": "Engineering"
        }

        return {
            "is_real": True,
            "matched": True,
            "liveness_score": liveness_score,
            "user": matched_user,
            "similarity_score": 0.92
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # The AI service runs on a different port (8001) to keep the web API (8000) free
    uvicorn.run(app, host="0.0.0.0", port=8001)
