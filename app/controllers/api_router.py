# Create the End-point predict/
# Do the calculation of Area, Location, date
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import numpy as np
import base64
import cv2
from datetime import date
from app.controllers.satellite_service import get_tif_files_array_from_MPC
from app.models.unet_inference import WaterSegmentationModel
from app.core.config import settings
from sqlalchemy.orm import Session
from app.models.database import get_db
from app.core.db_setup import WaterPredictions
from datetime import date
from app.models.schemas import PredicitonRequest
import math
import calendar
from datetime import date, timedelta
# Create a basemodel that make date option is dynamic so we can select the date we want to search for
class WaterPredictionQuery(BaseModel):
    bbox: list[float]
    start_date: date 
    end_date: date
print("Initalizating AI Engine inside API Router...")
ai_engine = WaterSegmentationModel(model_path=settings.MODEL_PATH)

router = APIRouter()


# Create a function that calculate the area of the water in the segmented image
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


def generate_dynamic_intervals(start_date: date, end_date: date):
    """
    Dynamically chops a date range into intervals.
    - If > 31 days: Chops into 1-month intervals.
    - If <= 31 days: Chops into 5-day intervals (matching Sentinel-2's orbit).
    Example: 2023-01-01 to 2023-03-15 becomes:
    [(2023-01-01, 2023-01-31), (2023-02-01, 2023-02-28), (2023-03-01, 2023-03-15)]
    Args:
    start_date: The absolute earliest date you want to search for where the user can select the date they want to search for
    end_date: The absolute latest date you want to search for where the user can select the
    """
    intervals=[] # List of tuples that contain the start and end date of each month in the range
    current_start = start_date
    total_days = (end_date - start_date).days

    if total_days <= 31:
        while current_start < end_date:
            # Find the next 5-day interval
            current_end = current_start + timedelta(days=4)
            if current_end > end_date:
                current_end = end_date
            
            intervals.append((current_start, current_end))
            # Move to the next interval
            current_start = current_end + timedelta(days=1)
    else:
        while current_start < end_date:
            # Find the last day of the current month
            _, last_day = calendar.monthrange(current_start.year, current_start.month) # Calculate the lenght of the month to get the last day of the month
            current_end = date(current_start.year, current_start.month, last_day) # Create a date object for the last day of the month
            if current_end > end_date:
                current_end = end_date
            
            intervals.append((current_start, current_end))

            # Move forward to the first day of the next month
            current_start = current_end + timedelta(days=1) # timedelta is used to add one day to the current end date to get the start date of the next month

    return intervals    
@router.post("/predict")
async def run_prediction_and_save(query: PredicitonRequest, db: Session = Depends(get_db)):
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
    
        # Save the predictions to the database
        db_prediction = WaterPredictions(
            bbox=query.bbox,
            capture_date=capture_date,
            water_percentage= round(water_percentage,2),
            total_area= total_area,
            water_area= water_area
        )

        db.add(db_prediction)
        db.commit()
        db.refresh(db_prediction)
        return {
            "message": "Inference completed and saved to database",
            "data": db_prediction,
            "mask_image": mask_base64
        }
    except Exception as e:
        # If anything fails (like clouds blocking the satellite), return a clean error
        print(f"API Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# @router.post("/predictions/")
# def save_predictions(id:int, bbox: str, capture_date: date, water_percentage: float, total_area: float, water_area: float, db: Session = Depends(get_db)):
#     """Endpoint to save predictions to the database."""
#     new_prediction = WaterPredictions(id=id, bbox=bbox, capture_date=capture_date, water_percentage=water_percentage, total_area=total_area)
#     db.add(new_prediction)
#     db.commit()
#     db.refresh(new_prediction)
#     return new_prediction

# Create an endpoint to make predictions of the water timeline
@router.post("/predict_timeline")
async def run_timeline_predictions(query: PredicitonRequest, db: Session = Depends(get_db)):
    """
    Takes a date range, chops it into monthly chunks, runs the AI on each chunk,
    saves the results to the database, and returns a historical timeline.
    """
    try: 
        print(f"Received timeline request for bbox: {query.bbox} from {query.start_date} to {query.end_date}")

        # Get the intervals from the start date to the end date in monthly intervals
        date_intervals = generate_dynamic_intervals(query.start_date, query.end_date)
        timeline_results = []

        for start, end in date_intervals:
            try:
                print(f"Processing interval: {start} to {end}")
                # Get the the tif files from the coardinates(bbox) for each month
                normalized_image, capture_date = get_tif_files_array_from_MPC(query.bbox, start_date=start, end_date=end)
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
                # Get the Water Area in KM^2
                total_area, water_area = get_areas(query.bbox, water_percentage)

                # Save the predictions to the database
                db_prediction = WaterPredictions(
                    bbox=query.bbox,
                    capture_date=capture_date,
                    water_percentage= round(water_percentage,2),
                    total_area= total_area,
                    water_area= water_area
                )

                db.add(db_prediction)
                db.commit()
                db.refresh(db_prediction)

                # Append to timeline results
                timeline_results.append({
                    "capture_date": capture_date,
                    "water_percentage": round(water_percentage,2),
                    "total_area": total_area,
                    "water_area": water_area
                })
            except Exception as e:
                # If the API call fails (like clouds blocking the satellite),skip that month and continue with the next month
                print(f"Skipping {start} to {end}: {str(e)}")
                continue
        
        if not timeline_results:
            raise ValueError("No clear satellite images found for any month in this range.")
        
        return {
            "message": "Timeline generated successfully",
            "timeline_data": timeline_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))