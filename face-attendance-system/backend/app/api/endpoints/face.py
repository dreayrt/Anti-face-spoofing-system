from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional, Any
import json
import httpx
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.models.employee import Employee

router = APIRouter()

AI_SERVICE_URL = "http://localhost:8001/predict"
MIN_DESCRIPTOR_LENGTH = 128
MIN_REGISTER_SAMPLES = 5


class FaceFrame(BaseModel):
    image: str
    box: Optional[dict[str, Any]] = None
    descriptor: Optional[List[float]] = None
    quality_metrics: Optional[dict[str, Any]] = None


class FaceRecognitionRequest(BaseModel):
    image: str
    box: dict | None = None
    descriptor: Optional[List[float]] = None
    frames: Optional[List[FaceFrame]] = None
    vote_min_match: Optional[int] = None


class FaceRecognitionResponse(BaseModel):
    success: bool
    message: str
    reason: Optional[str] = None
    user: Optional[dict] = None
    liveness_score: Optional[float] = None
    similarity_score: Optional[float] = None
    best_distance: Optional[float] = None
    match_threshold: Optional[float] = None
    frames_used: Optional[int] = None
    matched_votes: Optional[int] = None
    vote_threshold: Optional[int] = None


class FaceLivenessRequest(BaseModel):
    image: str
    box: dict | None = None


class FaceLivenessResponse(BaseModel):
    success: bool
    is_real: bool
    liveness_score: float


class FaceSample(BaseModel):
    image: str
    descriptor: List[float]
    quality: Optional[dict[str, Any]] = None


class FaceRegisterRequest(BaseModel):
    id: str
    name: str
    image: Optional[str] = None
    descriptor: Optional[List[float]] = None
    face_image_base64: Optional[str] = None
    face_descriptor: Optional[List[float]] = None
    samples: Optional[List[FaceSample]] = None


class FaceRegisterResponse(BaseModel):
    success: bool
    message: str
    user_id: Optional[str] = None
    reason: Optional[str] = None
    sample_count: Optional[int] = None


def is_valid_descriptor(descriptor: Optional[List[float]]) -> bool:
    if not descriptor or len(descriptor) != MIN_DESCRIPTOR_LENGTH:
        return False

    try:
        return all(value is not None for value in descriptor)
    except TypeError:
        return False


def average_descriptors(descriptors: List[List[float]]) -> List[float]:
    if not descriptors:
        return []

    summed = [0.0] * MIN_DESCRIPTOR_LENGTH
    for descriptor in descriptors:
        for idx, value in enumerate(descriptor):
            summed[idx] += float(value)

    return [value / len(descriptors) for value in summed]


@router.post("/register", response_model=FaceRegisterResponse)
def register_employee(request: FaceRegisterRequest, db: Session = Depends(get_db)):
    """
    Register a new employee with multi-sample descriptors (recommended >= 5 samples).
    Backward-compatible with legacy single-image payload.
    """
    try:
        existing = db.query(Employee).filter(Employee.id == request.id).first()
        if existing:
            raise HTTPException(status_code=400, detail="Employee ID already exists")

        sample_payloads: List[FaceSample] = []

        if request.samples:
            sample_payloads = request.samples
            if len(sample_payloads) < MIN_REGISTER_SAMPLES:
                raise HTTPException(
                    status_code=422,
                    detail=f"At least {MIN_REGISTER_SAMPLES} face samples are required for registration",
                )
        else:
            descriptor = request.face_descriptor if request.face_descriptor is not None else request.descriptor
            image_base64 = request.face_image_base64 if request.face_image_base64 else request.image

            if not image_base64:
                raise HTTPException(status_code=422, detail="Face image is required for registration")

            if not is_valid_descriptor(descriptor):
                raise HTTPException(
                    status_code=422,
                    detail="A valid 128D face descriptor is required for registration",
                )

            sample_payloads = [
                FaceSample(image=image_base64, descriptor=descriptor, quality=None)
            ]

        normalized_samples = []
        sample_descriptors: List[List[float]] = []

        for idx, sample in enumerate(sample_payloads):
            if not sample.image:
                raise HTTPException(status_code=422, detail=f"Sample #{idx + 1} missing image")
            if not is_valid_descriptor(sample.descriptor):
                raise HTTPException(status_code=422, detail=f"Sample #{idx + 1} has invalid 128D descriptor")

            descriptor_values = [float(v) for v in sample.descriptor]
            sample_descriptors.append(descriptor_values)
            normalized_samples.append(
                {
                    "image": sample.image,
                    "descriptor": descriptor_values,
                    "quality": sample.quality or {},
                }
            )

        prototype = average_descriptors(sample_descriptors)
        if not is_valid_descriptor(prototype):
            raise HTTPException(status_code=422, detail="Failed to build prototype descriptor from samples")

        descriptor_blob = {
            "version": 2,
            "sample_count": len(normalized_samples),
            "prototype": prototype,
            "samples": normalized_samples,
        }

        new_employee = Employee(
            id=request.id,
            name=request.name,
            face_image_base64=normalized_samples[0]["image"],
            face_descriptor=json.dumps(descriptor_blob),
        )
        db.add(new_employee)
        db.commit()
        db.refresh(new_employee)

        print(
            f"[Register] Saved employee {new_employee.id} - {new_employee.name} "
            f"with {len(normalized_samples)} samples"
        )

        return FaceRegisterResponse(
            success=True,
            message="Employee registered successfully.",
            user_id=new_employee.id,
            reason="registered",
            sample_count=len(normalized_samples),
        )
    except HTTPException:
        raise
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/recognize", response_model=FaceRecognitionResponse)
async def recognize_face(request: FaceRecognitionRequest):
    """
    Forward recognition payload to AI service.
    Supports single-frame and multi-frame vote payload.
    """
    try:
        frames_payload = None
        if request.frames:
            frames_payload = [
                {
                    "image_base64": frame.image,
                    "box": frame.box,
                    "descriptor": frame.descriptor,
                    "quality_metrics": frame.quality_metrics,
                }
                for frame in request.frames
            ]

        async with httpx.AsyncClient() as client:
            ai_response = await client.post(
                AI_SERVICE_URL,
                json={
                    "image_base64": request.image,
                    "box": request.box,
                    "descriptor": request.descriptor,
                    "frames": frames_payload,
                    "vote_min_match": request.vote_min_match,
                },
                timeout=30.0,
            )

        if ai_response.status_code != 200:
            raise HTTPException(status_code=500, detail="AI Service returned an error")

        result = ai_response.json()
        reason = result.get("reason")

        return FaceRecognitionResponse(
            success=bool(result.get("matched")),
            message=result.get("message", "Unknown response from AI service."),
            reason=reason,
            user=result.get("user"),
            liveness_score=result.get("liveness_score"),
            similarity_score=result.get("similarity_score"),
            best_distance=result.get("best_distance"),
            match_threshold=result.get("match_threshold"),
            frames_used=result.get("frames_used"),
            matched_votes=result.get("matched_votes"),
            vote_threshold=result.get("vote_threshold"),
        )

    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=(
                "Cannot connect to AI Service at http://localhost:8001/predict. "
                "Make sure the AI service is running on port 8001."
            ),
        )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="AI Service timed out. The service may be overloaded.",
        )
    except HTTPException:
        raise
    except Exception as exc:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/detect-face")
async def detect_face(request: FaceRecognitionRequest):
    return {"success": True, "faces_detected": 1}


@router.post("/liveness", response_model=FaceLivenessResponse)
async def check_liveness(request: FaceLivenessRequest):
    try:
        async with httpx.AsyncClient() as client:
            ai_response = await client.post(
                "http://localhost:8001/liveness",
                json={
                    "image_base64": request.image,
                    "box": request.box,
                },
                timeout=30.0,
            )

        if ai_response.status_code != 200:
            raise HTTPException(status_code=500, detail="AI Service returned an error")

        result = ai_response.json()
        return FaceLivenessResponse(
            success=True,
            is_real=result.get("is_real", False),
            liveness_score=result.get("liveness_score", 0.0),
        )

    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to AI Service. Make sure it is running.",
        )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="AI Service timed out.",
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
