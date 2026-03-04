from datetime import date
from pydantic import BaseModel, ConfigDict



# Define the Expected input data structure using pydantic
# Basemodel is used to validate the input data and make sure it is in the correct format, we can also add more fields if we want to like location name, timestamp, etc.
# Basemodel is a pydantic model 
class PredicitonRequest(BaseModel):
    bbox: list[float]
    start_date: date
    end_date: date
# Define what DB return after the prediction is made, we can also add more fields if we want to like location name, timestamp, etc.
class PredictionData(BaseModel):
    id: int
    bbox: str
    capture_date: date 
    water_percentage: float
    total_area: float
    water_area: float

    model_config = ConfigDict(from_attributes=True) # Pydantic only understands Python dictionaries. But SQLAlchemy returns custom "Database Objects". This line tells Pydantic, "Hey, it is okay to read data directly from a SQLAlchemy database row.


class PredictionResponse(BaseModel):
    message: str
    data: PredictionData
    mask_image: str