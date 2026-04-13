from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os

# Ideally, load this from python-dotenv or pydantic BaseSettings
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://postgres:123456@localhost:5432/FaceDetect"
)

# Create the SQLAlchemy engine connecting to the PostgreSQL database
engine = create_engine(
    DATABASE_URL,
    # pool_pre_ping=True helps handle dropped connections 
    pool_pre_ping=True,
    # echo=True is helpful for debugging queries
    echo=False
)

# Create a configured "Session" class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Declarative base for the ORM models
Base = declarative_base()

# Dependency to get a database session and safely close it
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
