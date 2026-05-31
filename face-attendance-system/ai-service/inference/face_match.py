from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any
import numpy as np
import json
import base64
import cv2
import os
import time

from mock_model import MockAntiSpoofModel, MockFaceRecognitionModel

app = FastAPI(title="AI Inference Service")

CHECKPOINT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "weights", "antispoof_cnn_dsp_lstm.pth"
)

LIVENESS_THRESHOLD = float(os.getenv("LIVENESS_THRESHOLD", "0.55"))
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.55"))
MIN_DESCRIPTOR_LENGTH = 128
CACHE_TTL_SECONDS = float(os.getenv("EMP_CACHE_TTL_SECONDS", "20"))
TOP_K_PROTOTYPES = int(os.getenv("TOP_K_PROTOTYPES", "5"))
MIN_VOTE_MATCH = int(os.getenv("MIN_VOTE_MATCH", "3"))

try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except Exception:
    faiss = None
    FAISS_AVAILABLE = False

spoof_model = None
USE_REAL_MODEL = False

try:
    from antispoof_model import AntiSpoofPredictor

    if os.path.isfile(CHECKPOINT_PATH):
        spoof_model = AntiSpoofPredictor(checkpoint_path=CHECKPOINT_PATH)
        USE_REAL_MODEL = True
        print("[AI Service] Loaded REAL Anti-Spoofing Model (CNN+DSP+LSTM)")
    else:
        raise RuntimeError(
            f"[AI Service] CRITICAL: Anti-spoof checkpoint not found: {CHECKPOINT_PATH}. "
            f"Cannot start without a real model. Please train the model first using train.py."
        )
except ImportError as exc:
    raise RuntimeError(
        f"[AI Service] CRITICAL: Cannot import AntiSpoofPredictor: {exc}. "
        f"Please ensure all dependencies (torch, torchvision, etc.) are installed."
    )

recognition_model = MockFaceRecognitionModel(weights_path="models/weights/facenet_model.pt")

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "dbname": os.getenv("DB_NAME", "FaceDetect"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "123456"),
}


class FrameRequest(BaseModel):
    image_base64: str
    box: dict | None = None
    descriptor: Optional[List[float]] = None
    quality_metrics: Optional[dict[str, Any]] = None


class InferenceRequest(BaseModel):
    image_base64: str
    box: dict | None = None
    descriptor: Optional[List[float]] = None
    frames: Optional[List[FrameRequest]] = None
    vote_min_match: Optional[int] = None


class LivenessRequest(BaseModel):
    image_base64: str
    box: dict | None = None


_EMP_CACHE = {
    "loaded_at": 0.0,
    "records": [],
    "record_by_id": {},
    "proto_vectors": None,
    "proto_ids": [],
    "ann_index": None,
}


def decode_base64_to_image(base64_str: str):
    if "," in base64_str:
        base64_data = base64_str.split(",", 1)[1]
    else:
        base64_data = base64_str
    img_data = base64.b64decode(base64_data)
    np_arr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def is_valid_descriptor(descriptor: Optional[List[float]]) -> bool:
    if not descriptor or len(descriptor) != MIN_DESCRIPTOR_LENGTH:
        return False

    try:
        return all(value is not None for value in descriptor)
    except TypeError:
        return False


def normalize_descriptor(descriptor: List[float]) -> List[float]:
    return [float(value) for value in descriptor]


def average_descriptors(descriptors: List[List[float]]) -> Optional[List[float]]:
    if not descriptors:
        return None

    matrix = np.array(descriptors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != MIN_DESCRIPTOR_LENGTH:
        return None

    return matrix.mean(axis=0).astype(np.float32).tolist()


def parse_face_descriptor_blob(raw_descriptor: Optional[str], fallback_image: Optional[str]):
    if not raw_descriptor:
        return [], None

    try:
        parsed = json.loads(raw_descriptor)
    except Exception:
        return [], None

    samples = []
    prototype = None

    if isinstance(parsed, list):
        if is_valid_descriptor(parsed):
            descriptor = normalize_descriptor(parsed)
            samples.append({
                "image": fallback_image,
                "descriptor": descriptor,
                "quality": {},
            })
            prototype = descriptor
        return samples, prototype

    if not isinstance(parsed, dict):
        return [], None

    raw_samples = parsed.get("samples") if isinstance(parsed.get("samples"), list) else []
    for sample in raw_samples:
        if not isinstance(sample, dict):
            continue

        descriptor = sample.get("descriptor")
        if not is_valid_descriptor(descriptor):
            continue

        samples.append(
            {
                "image": sample.get("image") or fallback_image,
                "descriptor": normalize_descriptor(descriptor),
                "quality": sample.get("quality") if isinstance(sample.get("quality"), dict) else {},
            }
        )

    raw_prototype = parsed.get("prototype")
    if is_valid_descriptor(raw_prototype):
        prototype = normalize_descriptor(raw_prototype)
    else:
        prototype = average_descriptors([sample["descriptor"] for sample in samples])

    return samples, prototype


def get_registered_employees_raw():
    import psycopg2

    conn = psycopg2.connect(**DB_CONFIG)
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, name, face_descriptor, face_image_base64 FROM employees")
        rows = cur.fetchall()
        cur.close()
        return rows
    finally:
        conn.close()


def build_ann_index(vectors: np.ndarray):
    if not FAISS_AVAILABLE:
        return None

    if vectors.size == 0:
        return None

    dim = vectors.shape[1]
    index = faiss.IndexHNSWFlat(dim, 32)
    index.hnsw.efConstruction = 40
    index.hnsw.efSearch = 64
    index.add(vectors.astype(np.float32))
    return index


def rebuild_employee_cache(force_reload: bool = False):
    now = time.time()
    if not force_reload and (now - _EMP_CACHE["loaded_at"] < CACHE_TTL_SECONDS):
        return _EMP_CACHE

    rows = get_registered_employees_raw()

    records = []
    record_by_id = {}
    proto_vectors = []
    proto_ids = []

    for employee_id, name, raw_descriptor, face_image_base64 in rows:
        samples, prototype = parse_face_descriptor_blob(raw_descriptor, face_image_base64)

        # Backward compatibility: if samples missing but prototype valid, create one sample from prototype
        if not samples and prototype and is_valid_descriptor(prototype):
            samples = [
                {
                    "image": face_image_base64,
                    "descriptor": normalize_descriptor(prototype),
                    "quality": {},
                }
            ]

        if not samples:
            continue

        if not prototype:
            prototype = average_descriptors([sample["descriptor"] for sample in samples])

        if not is_valid_descriptor(prototype):
            continue

        record = {
            "id": employee_id,
            "name": name,
            "face_image_base64": face_image_base64,
            "samples": samples,
            "prototype": normalize_descriptor(prototype),
        }
        records.append(record)
        record_by_id[employee_id] = record
        proto_vectors.append(record["prototype"])
        proto_ids.append(employee_id)

    if proto_vectors:
        proto_matrix = np.array(proto_vectors, dtype=np.float32)
        ann_index = build_ann_index(proto_matrix)
    else:
        proto_matrix = np.empty((0, MIN_DESCRIPTOR_LENGTH), dtype=np.float32)
        ann_index = None

    _EMP_CACHE["loaded_at"] = now
    _EMP_CACHE["records"] = records
    _EMP_CACHE["record_by_id"] = record_by_id
    _EMP_CACHE["proto_vectors"] = proto_matrix
    _EMP_CACHE["proto_ids"] = proto_ids
    _EMP_CACHE["ann_index"] = ann_index

    print(
        f"[FaceCache] Loaded {len(records)} employees, "
        f"ANN={'on' if ann_index is not None else 'off'}"
    )

    return _EMP_CACHE


def euclidean_distance(desc1, desc2):
    a = np.array(desc1, dtype=np.float32)
    b = np.array(desc2, dtype=np.float32)
    return float(np.linalg.norm(a - b))


def stage1_candidate_ids(query_descriptor: List[float], cache_data) -> List[str]:
    proto_vectors = cache_data["proto_vectors"]
    proto_ids = cache_data["proto_ids"]

    if proto_vectors is None or len(proto_ids) == 0:
        return []

    top_k = min(TOP_K_PROTOTYPES, len(proto_ids))
    if top_k <= 0:
        return []

    query = np.array(query_descriptor, dtype=np.float32).reshape(1, -1)

    ann_index = cache_data.get("ann_index")
    if ann_index is not None:
        distances, indices = ann_index.search(query, top_k)
        candidate_ids = []
        for idx in indices[0]:
            if idx < 0 or idx >= len(proto_ids):
                continue
            candidate_ids.append(proto_ids[idx])
        return candidate_ids

    # Fallback numpy exhaustive on prototypes
    dists = np.linalg.norm(proto_vectors - query[0], axis=1)
    top_indices = np.argsort(dists)[:top_k]
    return [proto_ids[int(i)] for i in top_indices]


def stage2_match(query_descriptor: List[float], cache_data):
    candidate_ids = stage1_candidate_ids(query_descriptor, cache_data)
    if not candidate_ids:
        return None, None

    best_record = None
    smallest_distance = float("inf")

    for employee_id in candidate_ids:
        record = cache_data["record_by_id"].get(employee_id)
        if not record:
            continue

        for sample in record["samples"]:
            descriptor = sample.get("descriptor")
            if not is_valid_descriptor(descriptor):
                continue

            distance = euclidean_distance(query_descriptor, descriptor)
            if distance < smallest_distance:
                smallest_distance = distance
                best_record = record

    return best_record, smallest_distance


def crop_face(img, box: Optional[dict]):
    if box is None:
        return img

    h_img, w_img, _ = img.shape

    x = max(0, int(box.get("x", 0)))
    y = max(0, int(box.get("y", 0)))

    # Support both old keys (w/h) and new keys (width/height)
    w = int(box.get("w", box.get("width", w_img)))
    h = int(box.get("h", box.get("height", h_img)))

    if w <= 0 or h <= 0:
        return img

    margin_x = int(w * 0.25)
    margin_y = int(h * 0.25)
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(w_img, x + w + margin_x)
    y2 = min(h_img, y + h + margin_y)

    face_crop = img[y1:y2, x1:x2]
    if face_crop.size == 0:
        return img
    return face_crop


def evaluate_single_frame(frame_payload, cache_data, sequence_liveness=None):
    image_base64 = frame_payload.get("image_base64")
    box = frame_payload.get("box")
    descriptor = frame_payload.get("descriptor")

    img = decode_base64_to_image(image_base64)
    if img is None:
        return {
            "is_real": False,
            "matched": False,
            "reason": "invalid_image",
            "message": "Invalid image data",
            "liveness_score": 0.0,
            "similarity_score": None,
            "best_distance": None,
            "user": None,
        }

    face_crop = crop_face(img, box)
    
    if sequence_liveness is not None:
        liveness_score = sequence_liveness
    else:
        liveness_score = float(spoof_model.predict_with_tta(face_crop))
        
    is_real = liveness_score > LIVENESS_THRESHOLD

    if not is_real:
        return {
            "is_real": False,
            "matched": False,
            "reason": "spoof_failed",
            "message": "Liveness check failed.",
            "liveness_score": liveness_score,
            "similarity_score": None,
            "best_distance": None,
            "user": None,
        }

    if not is_valid_descriptor(descriptor):
        return {
            "is_real": True,
            "matched": False,
            "reason": "descriptor_missing",
            "message": "No valid 128D face descriptor provided by client.",
            "liveness_score": liveness_score,
            "similarity_score": None,
            "best_distance": None,
            "user": None,
        }

    best_record, smallest_distance = stage2_match(descriptor, cache_data)

    if best_record is None or smallest_distance is None:
        return {
            "is_real": True,
            "matched": False,
            "reason": "no_registered_faces",
            "message": "No registered employees were found in the cache.",
            "liveness_score": liveness_score,
            "similarity_score": None,
            "best_distance": None,
            "user": None,
        }

    if smallest_distance >= MATCH_THRESHOLD:
        return {
            "is_real": True,
            "matched": False,
            "reason": "no_match",
            "message": "Face not recognized. Please retry in better lighting.",
            "liveness_score": liveness_score,
            "similarity_score": round(max(0.0, 1.0 - smallest_distance), 4),
            "best_distance": round(float(smallest_distance), 4),
            "user": None,
        }

    matched_user = {
        "id": best_record["id"],
        "name": best_record["name"],
        "face_image_base64": best_record.get("face_image_base64"),
    }

    return {
        "is_real": True,
        "matched": True,
        "reason": "recognized",
        "message": "Face recognized successfully.",
        "liveness_score": liveness_score,
        "similarity_score": round(max(0.0, 1.0 - smallest_distance), 4),
        "best_distance": round(float(smallest_distance), 4),
        "user": matched_user,
    }


def aggregate_vote_results(frame_results: List[dict], vote_min_match: Optional[int]):
    if not frame_results:
        return {
            "is_real": False,
            "matched": False,
            "reason": "no_frames",
            "message": "No frames were provided for recognition.",
            "liveness_score": None,
            "similarity_score": None,
            "best_distance": None,
            "user": None,
            "frames_used": 0,
            "matched_votes": 0,
            "vote_threshold": 0,
            "match_threshold": MATCH_THRESHOLD,
        }

    real_frames = [result for result in frame_results if result.get("is_real")]
    matched_frames = [result for result in frame_results if result.get("matched") and result.get("user")]

    frames_used = len(frame_results)
    vote_threshold = vote_min_match if vote_min_match is not None else max(MIN_VOTE_MATCH, (frames_used + 1) // 2)

    if not real_frames:
        return {
            "is_real": False,
            "matched": False,
            "reason": "spoof_failed",
            "message": "Liveness check failed on all frames.",
            "liveness_score": round(float(np.mean([result.get("liveness_score", 0.0) for result in frame_results])), 4),
            "similarity_score": None,
            "best_distance": None,
            "user": None,
            "frames_used": frames_used,
            "matched_votes": 0,
            "vote_threshold": vote_threshold,
            "match_threshold": MATCH_THRESHOLD,
        }

    # Count votes by employee id
    vote_counter = {}
    for item in matched_frames:
        user = item.get("user") or {}
        employee_id = user.get("id")
        if not employee_id:
            continue
        vote_counter[employee_id] = vote_counter.get(employee_id, 0) + 1

    if not vote_counter:
        return {
            "is_real": True,
            "matched": False,
            "reason": "no_match",
            "message": "Access denied. Face did not match registered employees.",
            "liveness_score": round(float(np.mean([result.get("liveness_score", 0.0) for result in real_frames])), 4),
            "similarity_score": None,
            "best_distance": min([result.get("best_distance") for result in real_frames if result.get("best_distance") is not None], default=None),
            "user": None,
            "frames_used": frames_used,
            "matched_votes": 0,
            "vote_threshold": vote_threshold,
            "match_threshold": MATCH_THRESHOLD,
        }

    best_employee_id = max(vote_counter, key=vote_counter.get)
    matched_votes = vote_counter[best_employee_id]

    if matched_votes < vote_threshold:
        return {
            "is_real": True,
            "matched": False,
            "reason": "vote_not_enough",
            "message": "Access denied. Not enough stable match votes.",
            "liveness_score": round(float(np.mean([result.get("liveness_score", 0.0) for result in real_frames])), 4),
            "similarity_score": None,
            "best_distance": min([result.get("best_distance") for result in matched_frames if result.get("best_distance") is not None], default=None),
            "user": None,
            "frames_used": frames_used,
            "matched_votes": matched_votes,
            "vote_threshold": vote_threshold,
            "match_threshold": MATCH_THRESHOLD,
        }

    winner_frames = [
        result for result in matched_frames if (result.get("user") or {}).get("id") == best_employee_id
    ]
    winner_user = winner_frames[0].get("user") if winner_frames else None

    similarity_values = [result.get("similarity_score") for result in winner_frames if result.get("similarity_score") is not None]
    distance_values = [result.get("best_distance") for result in winner_frames if result.get("best_distance") is not None]

    return {
        "is_real": True,
        "matched": True,
        "reason": "recognized",
        "message": "Face recognized successfully.",
        "liveness_score": round(float(np.mean([result.get("liveness_score", 0.0) for result in real_frames])), 4),
        "similarity_score": round(float(np.mean(similarity_values)), 4) if similarity_values else None,
        "best_distance": round(float(min(distance_values)), 4) if distance_values else None,
        "user": winner_user,
        "frames_used": frames_used,
        "matched_votes": matched_votes,
        "vote_threshold": vote_threshold,
        "match_threshold": MATCH_THRESHOLD,
    }


@app.post("/predict")
async def predict_face(request: InferenceRequest):
    try:
        cache_data = rebuild_employee_cache(force_reload=False)
        if not cache_data["records"]:
            return {
                "is_real": True,
                "matched": False,
                "reason": "no_registered_faces",
                "message": "No registered employees were found in the database.",
                "liveness_score": None,
                "user": None,
                "match_threshold": MATCH_THRESHOLD,
                "frames_used": 0,
                "matched_votes": 0,
                "vote_threshold": 0,
            }

        if request.frames and len(request.frames) > 0:
            frame_payloads = []
            face_crops = []
            
            for frame in request.frames:
                payload = {
                    "image_base64": frame.image_base64,
                    "box": frame.box,
                    "descriptor": frame.descriptor,
                    "quality_metrics": frame.quality_metrics,
                }
                frame_payloads.append(payload)
                
                # Extract crop for video sequence
                img = decode_base64_to_image(frame.image_base64)
                if img is not None:
                    face_crops.append(crop_face(img, frame.box))
            
            # 1. Temporal Liveness Check (LSTM Multi-frame)
            if hasattr(spoof_model, 'predict_video') and len(face_crops) > 0:
                sequence_liveness = float(spoof_model.predict_video(face_crops))
            else:
                # Fallback to single frame if predict_video is not available
                sequence_liveness = float(spoof_model.predict_with_tta(face_crops[-1])) if face_crops else 0.0

            # 2. Face Recognition + Format Results
            frame_results = [
                evaluate_single_frame(frame_payload, cache_data, sequence_liveness=sequence_liveness) 
                for frame_payload in frame_payloads
            ]
        else:
            frame_payloads = [
                {
                    "image_base64": request.image_base64,
                    "box": request.box,
                    "descriptor": request.descriptor,
                    "quality_metrics": None,
                }
            ]
            frame_results = [evaluate_single_frame(frame_payload, cache_data) for frame_payload in frame_payloads]

        aggregated = aggregate_vote_results(frame_results, request.vote_min_match)
        return aggregated

    except Exception as exc:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/liveness")
async def check_liveness(request: LivenessRequest):
    try:
        img = decode_base64_to_image(request.image_base64)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
            
        face_crop = crop_face(img, request.box)
        liveness_score = float(spoof_model.predict_with_tta(face_crop))
        is_real = liveness_score > LIVENESS_THRESHOLD
        
        return {
            "success": True,
            "is_real": is_real,
            "liveness_score": round(liveness_score, 4)
        }
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/refresh-cache")
async def refresh_cache():
    try:
        cache_data = rebuild_employee_cache(force_reload=True)
        return {
            "success": True,
            "employees": len(cache_data["records"]),
            "ann_enabled": bool(cache_data.get("ann_index") is not None),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
