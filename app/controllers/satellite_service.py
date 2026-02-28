# Take the (bbox) Coardinates (x_min,x_max,y_min,y_max) from the fast api send 
# a Get request to the MPC search in it's DB for these coardinates---> response with the file.tif 
from rasterio.env import Env
from pystac_client import Client
import planetary_computer
import rasterio
import numpy as np
import cv2
from rasterio.warp import transform_bounds

def get_tif_files_array_from_MPC(bbox: list,start_date: str, end_date: str,  target_shape=(128, 128)):
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
        datetime=f"{start_date}/{end_date}", # Search the last year
        query={"eo:cloud_cover": {"lt": 5}}, # Less than 5% clouds
        max_items=1
    )
    items = list(search.items())
    
    if not items:
        raise ValueError("No clear satellite images found for this location.")
    
    best_item = items[0]
    print(f"Found image from date: {best_item.datetime}")

    # The U-Net was trained on a 12-band dataset containing geographical layers:
    # 0: Aerosol (B01), 1: Blue (B02), 2: Green (B03), 3: Red (B04), 4: NIR (B08)
    # 5: SWIR1 (B11), 6: SWIR2 (B12), 7: QA Mask, 8: DEM1, 9: DEM2, 10: ESA WorldCover, 11: JRC Water Probability
    optical_bands = [
        "B01", "B02", "B03", "B04", "B08", "B11", "B12"
    ]
    
    optical_arrays = []
    print(f"📡 Starting download of Sentinel-2 optical bands...")

    # The Env block optimizes rasterio for lightning-fast cloud reading for fast loading the data from MPC
    with Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', 
             CPL_VSIL_CURL_ALLOWED_EXTENSIONS='tif', 
             VSI_CACHE=True):
        # 3. Download and process each optical band
        for band in optical_bands:
            print(f"   ⬇️ Downloading {band}...") 
            href = best_item.assets[band].href
            with rasterio.open(href) as src:
                utm_bbox = transform_bounds("EPSG:4326", src.crs, *bbox)
                window = rasterio.windows.from_bounds(*utm_bbox, transform=src.transform)
                band_data = src.read(1, window=window)
                
                if band_data.size == 0:
                    raise ValueError("The requested bounding box is out of bounds for this satellite image.")
                
                band_resized = cv2.resize(
                    band_data, target_shape, interpolation=cv2.INTER_LINEAR
                )
                optical_arrays.append(band_resized)

    # Calculate NDWI to synthesize the geographic probability bands
    # NDWI = (Green - NIR) / (Green + NIR)
    green = optical_arrays[2].astype(np.float32)
    nir = optical_arrays[4].astype(np.float32)
    ndwi = (green - nir) / (green + nir + 1e-8)
    
    # 7: QA Band (all zeros)
    qa_band = np.zeros(target_shape, dtype=np.uint16)
    # 8, 9: DEMs (all zeros for flat terrain)
    dem1 = np.zeros(target_shape, dtype=np.uint16)
    dem2 = np.zeros(target_shape, dtype=np.uint16)
    
    # 10: ESA WorldCover (10 for land, 80 for water)
    esa_band = np.where(ndwi > 0, 80, 10).astype(np.uint16)
    
    # 11: JRC Water Occurrence Probability (0 to 100)
    jrc_band = np.where(ndwi > 0, 100, 0).astype(np.uint16)

    # Combine optical and synthetic geographic channels perfectly matching training
    final_bands = optical_arrays + [qa_band, dem1, dem2, esa_band, jrc_band]
    
    # 4. Stack into a single 3D array: shape becomes (128, 128, 12)
    stacked_image = np.dstack(final_bands).astype(np.float32)
    
    # 5. Apply the exact same Min-Max Normalization used during your model training
    normalized_image = (stacked_image - np.min(stacked_image)) / (np.max(stacked_image) - np.min(stacked_image) + 1e-8)
    
    # Extract the date from best_item and return BOTH!
    capture_date = best_item.datetime.strftime("%Y-%m-%d")
        
    return normalized_image, capture_date


