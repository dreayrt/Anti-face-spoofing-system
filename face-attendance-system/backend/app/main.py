from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import face
from app.database.session import engine, Base
from app.models.employee import Employee # Ensure model is imported so tables are created

# Create Database Tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Face Attendance API",
    description="Backend API for Face Attendance & Anti-Spoofing System",
    version="1.0.0"
)

# Configure CORS so the React frontend can communicate with the FastAPI backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"], # React default ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the endpoints router
# In a real app you'd likely have a main router.py that includes multiple endpoint files
app.include_router(face.router, prefix="/api/v1/face", tags=["Face Recognition"])

@app.get("/api/v1/health", tags=["System"])
def health_check():
    """Health check endpoint to ensure the server is running."""
    return {"status": "ok", "message": "Face Attendance API is running"}

