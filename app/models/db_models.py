# # We will use PostgreSQL to store a permanent history of your analyses.
# #  This allows you to track how water levels change in the Nile or Lake Nasser over time.

# from sqlalchemy import Column, Integer, Float, String, DateTime, JSON
# from sqlalchemy.ext.declarative import declarative_base
# import datetime

# Base = declarative_base()

# class WaterAnalysisRecord(Base):
#     __tablename__ = "analysis_history"

#     id = Column(Integer, primary_key=True, index=True)
#     capture_date = Column(String)     # The date from the satellite
#     processed_at = Column(DateTime, default=datetime.datetime.utcnow)
#     bbox = Column(JSON)               # Stored as [min_lon, min_lat, max_lon, max_lat]
#     total_area_km2 = Column(Float)    # Total scan area
#     water_area_km2 = Column(Float)    # Calculated water area
#     water_percentage = Column(Float)  # Percentage coverage