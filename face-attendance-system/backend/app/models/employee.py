from sqlalchemy import Column, String, Text, DateTime
from app.database.session import Base
from datetime import datetime

class Employee(Base):
    __tablename__ = "employees"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, index=True)
    face_image_base64 = Column(Text, nullable=True)
    face_descriptor = Column(Text, nullable=True)  # 128D face descriptor as JSON array
    created_at = Column(DateTime, default=datetime.utcnow)

