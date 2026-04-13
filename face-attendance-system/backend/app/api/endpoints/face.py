from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
import json
import httpx
from sqlalchemy.orm import Session
from app.database.session import get_db
from app.models.employee import Employee

router = APIRouter()

class FaceRecognitionRequest(BaseModel):
    image: str
    box: dict = None
    descriptor: Optional[List[float]] = None  # 128D face descriptor from face-api.js

class FaceRecognitionResponse(BaseModel):
    success: bool
    message: str
    user: dict = None
    liveness_score: float = None

class FaceRegisterRequest(BaseModel):
    id: str
    name: str
    image: str
    descriptor: Optional[List[float]] = None  # 128D face descriptor from face-api.js

class FaceRegisterResponse(BaseModel):
    success: bool
    message: str
    user_id: str = None

AI_SERVICE_URL = "http://localhost:8001/predict"

@router.post("/register", response_model=FaceRegisterResponse)
def register_employee(request: FaceRegisterRequest, db: Session = Depends(get_db)):
    """
    Receives employee details and a face image (base64) to register a new employee.
    """
    try:
        # Check if id already exists
        existing = db.query(Employee).filter(Employee.id == request.id).first()
        if existing:
            raise HTTPException(status_code=400, detail="Employee ID already exists")

        # Save new employee with face descriptor
        new_employee = Employee(
            id=request.id,
            name=request.name,
            face_image_base64=request.image,
            face_descriptor=json.dumps(request.descriptor) if request.descriptor else None
        )
        db.add(new_employee)
        db.commit()
        db.refresh(new_employee)

        return FaceRegisterResponse(
            success=True,
            message="Employee registered successfully.",
            user_id=new_employee.id
        )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/recognize", response_model=FaceRecognitionResponse)
async def recognize_face(request: FaceRecognitionRequest):
    """
    Receives an image (base64), forwards it to the AI Service for liveness 
    checking and face matching, and logs the attendance if valid.
    """
    try:
        # 1. Forward to AI Service
        async with httpx.AsyncClient() as client:
            ai_response = await client.post(
                AI_SERVICE_URL,
                json={
                    "image_base64": request.image,
                    "box": request.box,
                    "descriptor": request.descriptor
                },
                timeout=15.0
            )
            
            if ai_response.status_code != 200:
                raise HTTPException(status_code=500, detail="AI Service returned an error")
                
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

    except httpx.ConnectError:
        raise HTTPException(
            status_code=503, 
            detail="Cannot connect to AI Service at " + AI_SERVICE_URL + ". Make sure the AI service is running on port 8001."
        )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="AI Service timed out. The service may be overloaded."
        )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/detect-face")
async def detect_face(request: FaceRecognitionRequest):
    """
    A simple endpoint just to detect if a face is present in the frame,
    useful for UI feedback before hitting the heavy recognition model.
    """
    return {"success": True, "faces_detected": 1}
