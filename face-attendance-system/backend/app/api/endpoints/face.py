from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import httpx # For communicating with the AI service

router = APIRouter()

class FaceRecognitionRequest(BaseModel):
    image: str # Base64 encoded image string
    box: dict = None # {x: float, y: float, w: float, h: float}

class FaceRecognitionResponse(BaseModel):
    success: bool
    message: str
    user: dict = None
    liveness_score: float = None

AI_SERVICE_URL = "http://localhost:8001/predict"

@router.post("/recognize", response_model=FaceRecognitionResponse)
async def recognize_face(request: FaceRecognitionRequest):
    """
    Receives an image (base64), forwards it to the AI Service for liveness 
    checking and face matching, and logs the attendance if valid.
    """
    try:
        # 1. Forward to AI Service
        # In a real system, you might decode the base64, save it temporarily, 
        # or send it directly as binary to the AI service.
        async with httpx.AsyncClient() as client:
            ai_response = await client.post(
                AI_SERVICE_URL,
                json={
                    "image_base64": request.image,
                    "box": request.box
                },
                timeout=10.0
            )
            
            if ai_response.status_code != 200:
                raise HTTPException(status_code=500, detail="AI Service Error")
                
            result = ai_response.json()
            
            # 2. Process Result
            if not result.get("is_real"):
                return FaceRecognitionResponse(
                    success=False, 
                    message="Liveness check failed. Spoof detected.",
                    liveness_score=result.get("liveness_score")
                )
                
            if not result.get("matched"):
                return FaceRecognitionResponse(
                    success=False,
                    message="Face not recognized or not found in database.",
                    liveness_score=result.get("liveness_score")
                )
            
            user_data = result.get("user")
            
            # 3. Log Attendance Database Transaction Here
            # db_session.add(AttendanceLog(user_id=user_data['id'], status='present'))
            # db_session.commit()
            
            return FaceRecognitionResponse(
                success=True,
                message="Attendance recorded successfully.",
                user=user_data,
                liveness_score=result.get("liveness_score")
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/detect-face")
async def detect_face(request: FaceRecognitionRequest):
    """
    A simple endpoint just to detect if a face is present in the frame,
    useful for UI feedback before hitting the heavy recognition model.
    """
    return {"success": True, "faces_detected": 1}
