from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# The connection URL matching your docker-compose credentials
SQLALCHEMY_DATABASE_URL = "postgresql://ai_user:securepassword123@localhost:5432/water_segmentation"

# Create the SQLAlchemy engine
engine = create_engine(SQLALCHEMY_DATABASE_URL)

# Create a session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for your database models
Base = declarative_base()

# Dependency function to get the database session in FastAPI
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()