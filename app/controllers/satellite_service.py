# Take the (bbox) Coardinates (x_min,x_max,y_min,y_max) from the fast api send 
# a Get request to the MPC search in it's DB for these coardinates---> response with the file.tif 
from rasterio.env import Env
from pystac_client import Client
import planetary_computer
import rasterio
import numpy as np
import cv2
from rasterio.warp import transform_bounds

def get_tif_files_array_from_MPC(bbox: list, target_shape=(128, 128)):
    """
    Queries Microsoft Planetary Computer for Sentinel-2 data over a specific GPS box,
    downloads the 12 required bands, and stacks them into a single array.
    
    Args:
        bbox (list): [min_lon, min_lat, max_lon, max_lat]
        target_shape (tuple): The spatial dimensions required by the U-Net.
        
    Returns:
        np.ndarray: A stacked, normalized array of shape (128, 128, 12) ready for inference.
    """
    print(f"Searching catalog for coordinates: {bbox}")
    
    # 1. Connect to Microsoft's open satellite database
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    # 2. Search for the newest cloud-free Sentinel-2 image over this bounding box
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime="2025-01-01/2025-12-31", # Search the last year
        query={"eo:cloud_cover": {"lt": 5}}, # Less than 5% clouds
        max_items=1
    )
    items = list(search.items())
    
    if not items:
        raise ValueError("No clear satellite images found for this location.")
    
    best_item = items[0]
    print(f"Found image from date: {best_item.datetime}")

    # The 12 exact bands Sentinel-2 uses (matching your training data)
    required_bands = [
        "B01", "B02", "B03", "B04", "B05", "B06", 
        "B07", "B08", "B8A", "B09", "B11", "B12"
    ]
    
    band_arrays = []
    print(f"📡 Starting download of 12 satellite bands...")

    # The Env block optimizes rasterio for lightning-fast cloud reading for fast loading the data from MPC
    with Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', 
             CPL_VSIL_CURL_ALLOWED_EXTENSIONS='tif', 
             VSI_CACHE=True):
        # 3. Download and process each band
        for band in required_bands:
            print(f"   ⬇️ Downloading {band}...") # This tells you it isn't frozen!
            href = best_item.assets[band].href
            with rasterio.open(href) as src:
                # Translate the Lat/Lon BBox into the Satellite's native UTM coordinates
                utm_bbox = transform_bounds("EPSG:4326", src.crs, *bbox)
                
                # Read the cropped window using the translated UTM coordinates
                window = rasterio.windows.from_bounds(*utm_bbox, transform=src.transform)
                band_data = src.read(1, window=window)
                
                # Safety check to prevent OpenCV from crashing if the array is empty
                if band_data.size == 0:
                    raise ValueError("The requested bounding box is out of bounds for this satellite image.")
                
                # Resize it strictly to 128x128 to satisfy the U-Net geometry
                band_resized = cv2.resize(
                    band_data, target_shape, interpolation=cv2.INTER_LINEAR
                )
                band_arrays.append(band_resized)

    # 4. Stack into a single 3D array: shape becomes (128, 128, 12)
    stacked_image = np.dstack(band_arrays).astype(np.float32)
    
    # 5. Apply the exact same Min-Max Normalization used during your model training
    normalized_image = (stacked_image - np.min(stacked_image)) / (np.max(stacked_image) - np.min(stacked_image) + 1e-8)
    # 🚨 THE FIX: Extract the date from best_item and return BOTH!
    capture_date = best_item.datetime.strftime("%Y-%m-%d")
        
    return normalized_image , capture_date


