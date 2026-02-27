import streamlit as st
import folium
from folium.plugins import Draw, Geocoder
from streamlit_folium import st_folium
import requests
import base64
import numpy as np
import cv2

# 1. Page Configuration
st.set_page_config(page_title="🌍 Water Segmentation AI", layout="wide")
st.title("🌍 Multispectral Water Segmentation")
st.markdown("Select a location, draw a bounding box, and let the U-Net extract the water bodies.")

# 🚨 THE DROPDOWN SEARCH: Type 2 letters to filter this list!
locations = {
    "Aswan Low Dam, Egypt": [24.0350, 32.8750],
    "Nile Delta, Egypt": [31.4000, 31.0000],
    "Lake Nasser, Egypt": [22.7500, 32.7500],
    "Lake Victoria, Africa": [-1.0000, 33.0000],
    "Lake Mead, USA": [36.0163, -114.7370]
}
selected_loc = st.selectbox("🔍 Quick Jump to Location (Type to search):", list(locations.keys()))
map_center = locations[selected_loc]

# 2. Initialize the Map
# m = folium.Map(location=map_center, zoom_start=12, tiles="CartoDB positron")
m = folium.Map(location=map_center, zoom_start=12, tiles="OpenStreetMap")
Geocoder().add_to(m)

# 3. Add the Drawing Tools
Draw(
    draw_options={'polyline': False, 'polygon': False, 'circle': False, 'marker': False, 'circlemarker': False, 'rectangle': True},
    edit_options={'edit': False}
).add_to(m)

# 4. Display the Map
st.subheader("Select Target Area")
map_output = st_folium(m, width=1000, height=500)

st.subheader("📡 AI Extraction")

# 5. Extract Coordinates and Connect to API
if map_output and map_output.get("last_active_drawing"):
    coordinates = map_output["last_active_drawing"]["geometry"]["coordinates"][0]
    lons = [p[0] for p in coordinates]
    lats = [p[1] for p in coordinates]
    bbox = [min(lons), min(lats), max(lons), max(lats)]
    
    st.success(f"The Selected Coordinates(BBox): `{bbox}`")
    
    # THE API CONNECTION BUTTON
    if st.button("Get the Segmented water body", type="primary"):
        
        # Streamlit spinner to show the user the AI is thinking
        with st.spinner("Downloading satellite data and running neural network..."):
            try:
                # 1. Send the data to your FastAPI backend
                api_url = "http://127.0.0.1:8000/api/predict"
                response = requests.post(api_url, json={"bbox": bbox})
                
                if response.status_code == 200:
                    data = response.json()
                    
                    st.markdown("---")
                    st.subheader("📊 Analytics Report")
                    
                    # 2. Display the stats in beautiful UI metric cards
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Capture Date", data["capture_date"])
                    col2.metric("Total Scan Area", f"{data['total_area_km^2']} km²")
                    col3.metric("Water Area", f"{data['water_area_km^2']} km²")
                    col4.metric("Water Coverage", f"{data['water_percentage']}%")
                    
                    # 3. THE UNBOXING: Base64 back to Image
                    img_bytes = base64.b64decode(data["mask_base64"])
                    mask_array = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
                    
                    # 4. The Computer Vision Trick: Turn White pixels into Blue water pixels!
                    blue_mask = np.zeros((128, 128, 3), dtype=np.uint8)
                    blue_mask[mask_array == 255] = [0, 150, 255] # RGB color for water
                    
                    # Display the final image
                    st.image(blue_mask, caption="AI Predicted Water Mask", width=400)
                    
                else:
                    st.error(f"Backend Error: {response.text}")
                    
            except Exception as e:
                st.error(f"Could not connect to FastAPI. Is your server running? Error: {e}")
else:
    st.info("Waiting for you to draw a rectangle on the map...")