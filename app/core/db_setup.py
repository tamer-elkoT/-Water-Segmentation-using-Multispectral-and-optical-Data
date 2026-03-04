from sqlalchemy import Column, Integer, String, Float, Date
from app.models.database import Base

class WaterPredictions(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    bbox = Column(String, nullable=False, index=True)  # Store as JSON string
    capture_date = Column(Date, nullable=False, index=True)
    water_percentage = Column(Float, nullable=False)
    total_area = Column(Float, nullable=False)
    water_area = Column(Float, nullable=False)
    # We Can add timstamp, location name, 


