# Create the End-point predict/
# Do the calculation of Area, Location, date
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
import base64
import cv2

from app.controllers.satellite_service import get_tif_files_array_from_MPC
from app.models.unet_inference import WaterSegmentationModel
from app.core.config import settings
# Create a basemodel that make date option is dynamic so we can select the date we want to search for
class WaterPredictionQuery(BaseModel):
    bbox: list[float]
    start_date: str 
    end_date: str
print("Initalizating AI Engine inside API Router...")
ai_engine = WaterSegmentationModel(model_path=settings.MODEL_PATH)

router = APIRouter()

# Define the Expected input data structure using pydantic
class LocationQuery(BaseModel):
    bbox: list[float]
    start_date: str
    end_date: str
# Create a function that calculate the area of the water in the segmented image
import math

def get_areas(bbox: list[float], water_percentage: float):
    """
    Converts a GPS bounding box into Square Kilometers, 
    and calculates how much of that area is covered by water.
    """
    # min_lon = x_min: The absolute Left edge of your bounding box.
    # max_lon = x_max: The absolute Right edge of your bounding box. 
    # min_lat = y_min: The absolute Bottom edge of your bounding box.
    # max_lat = y_max: The absolute Top edge of your bounding box.
    min_lon, min_lat, max_lon, max_lat = bbox

    # 1. Get the average latitude in radians (required for math.cos)
    avg_lat_rad = math.radians((min_lat + max_lat) / 2.0)
    
    # 2. Calculate the width and height of the box in Kilometers
    height_km = (max_lat - min_lat) * 111.32
    width_km = (max_lon - min_lon) * 111.32 * math.cos(avg_lat_rad)
    
    # 3. Calculate the areas
    total_area_sqkm = width_km * height_km
    water_area_sqkm = total_area_sqkm * (water_percentage / 100.0)

    return round(total_area_sqkm, 2), round(water_area_sqkm, 2)

@router.post("/predict")
async def predict_water(query: LocationQuery):
    """
    Receives a GPS bounding box, fetches live satellite data, runs the U-Net,
    and returns the water mask as a Base64 encoded image.
    """
    try:
        print(f"Received request for bbox: {query.bbox}")
        # Get the the tif files from the coardinates(bbox)
        normalized_image, capture_date = get_tif_files_array_from_MPC(query.bbox, start_date=query.start_date, end_date=query.end_date)
        # Run the U-Net Predictions
        mask = ai_engine.predict(normalized_image) # An Array (128,128) of 0s (land pixel) and 1s (watr pixel)
     
        binary_mask = mask.astype(np.uint8)
        # Calculate the percentage of area the existing water in the mask
        # if the pixels are water 
        water_pixels = np.count_nonzero(binary_mask==1)
        # get the total pixels water(1) + land(0)
        total_pixels = binary_mask.size
        # get the percentage of the water in the segmented image
        water_percentage = (water_pixels/total_pixels)*100
        # prepare the mask by multiply the mask by 255 to make it contain range from (0,255) instead of (0,1)
        mask_visual = (mask * 255).astype(np.uint8)
        # compress the numpy array in the mask into a .png file
        # success is True if open cv successfully creates the PNG file
        # buffer is 1 D numpy array of mask bytes
        # FastAPI uses OpenCV and Base64 to turn that numpy array into a long text string.
        success, buffer = cv2.imencode(".png", mask_visual)
        # If the success is false (cv2 failed to create the png file)
        if not success:
            raise ValueError("Failed to encode image mask.")
        mask_base64 = base64.b64encode(buffer).decode("utf-8")
        # Get the Water Area in KM^2
        total_area, water_area = get_areas(query.bbox, water_percentage)
        # Return the Jsong payload of the status of the image 
        return {
            "is open cv creates the png file": success,
            "water_percentage": round(water_percentage, 2),
            "total_area_km^2": total_area,
            "water_area_km^2": water_area,
            "mask_base64": mask_base64,
            "capture_date": capture_date

        }

    except Exception as e:
        # If anything fails (like clouds blocking the satellite), return a clean error
        print(f"API Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
