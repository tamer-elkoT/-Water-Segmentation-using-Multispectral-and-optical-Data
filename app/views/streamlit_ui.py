import streamlit as st
import folium
from folium.plugins import Draw, Geocoder, LocateControl
from streamlit_folium import st_folium
import requests
import base64
import numpy as np
import cv2
import math
import datetime
import pandas as pd
# 1. Page Configuration (Must be the very first Streamlit command)
st.set_page_config(
    page_title="AI Water Segmentation",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 🧭 SIDEBAR: INSTRUCTIONS & CONTACT INFO
# ==========================================
with st.sidebar:
    st.divider()
    st.header("📅 Time-Series Analysis")
    
    # Calculate default dates (Last 90 days up to today)
    today = datetime.date.today()
    ninety_days_ago = today - datetime.timedelta(days=90)
    
    # # Create the dynamic calendar widget
    # selected_dates = st.date_input(
    #     "Select Monitoring Period:",
    #     value=(ninety_days_ago, today), # Default selection
    #     max_value=today                 # Prevent picking future dates!
    # )
    col_start, col_end = st.columns(2)
    with col_start:
        start_date_obj = st.date_input("Start Date", value=ninety_days_ago, max_value=today)
    with col_end:
        end_date_obj = st.date_input("End Date", value=today, max_value=today)
    st.header("📖 How to Use")
    st.markdown("""
    1. **Search** for a location using the dropdown or the map's search icon.
    2. **Change Map View** using the layer icon (top right) to see satellite or street views.
    3. **Draw a Rectangle** over a body of water using the square icon on the left.
    4. **Run Analysis** to let the AI process the satellite data!
    """)
    
    # The Bounding Box Size Warning

    st.warning("""
    **📏 Bounding Box Guide:**
    - **Too Small (< 50m):** The satellite resolution (10m/pixel) cannot detect it.
    - **Too Big (> 10km):** Downloading massive satellite files will overload the network.
    - **Perfect Size:** Draw an area roughly 1 to 3 kilometers wide (the size of a large neighborhood).
    """, icon="⚠️")
    
    st.divider()
    
    # Author & Contact Info
    st.header("👨‍💻 About the Author")
    st.markdown("""
    **Tamer** *Full Stack AI Engineer* 📍 Cairo, Egypt  
    
    Built with Python, FastAPI, and a custom U-Net architecture to process live Sentinel-2 multispectral satellite data.
    """)
    
    st.markdown("### 📬 Support & Links")
    st.markdown("[📧 Email Support](tamer.elkot.ai@gmail.com)")
    st.markdown("[💼 LinkedIn](https://www.linkedin.com/in/tamer-elkot/)")
    st.markdown("[🐙 GitHub Repository](https://github.com/tamer-elkoT)")
    st.markdown("[📧 Portfolio](https://tamerelkot.netlify.app/)")
# 1. Page Configuration
st.set_page_config(page_title="🌍 Water Segmentation AI", layout="wide")
st.title("🌍 Multispectral Water Segmentation")
st.markdown("Draw a bounding box anywhere on Earth to instantly download live satellite data and map water bodies using Deep Learning.")

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
# Google Map UI
m = folium.Map(location=map_center, zoom_start=12, tiles="OpenStreetMap")
# Google Earth UI
# m = folium.Map(
#     location=map_center, 
#     zoom_start=12, 
#     tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
#     attr='Esri World Imagery'
# )
# Add Layer 1: Satellite View (Esri World Imagery)
folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr='Esri',
    name='Satellite View'
).add_to(m)

# Add Layer 2: Minimalist View (Best for making the AI blue mask pop!)
folium.TileLayer(
    tiles='CartoDB positron',
    name='Minimalist (AI View)'
).add_to(m)
#  Add the Layer Control button so the user can switch between them
folium.LayerControl().add_to(m)
Geocoder().add_to(m)
# Add the Locate Me GPS tracking button
LocateControl().add_to(m)
# Add the Drawing Tools
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
    
    # FRONTEND VALIDATION MATH ---
    avg_lat_rad = math.radians((bbox[1] + bbox[3]) / 2.0)
    height_km = (bbox[3] - bbox[1]) * 111.32
    width_km = (bbox[2] - bbox[0]) * 111.32 * math.cos(avg_lat_rad)
    area_sqkm = width_km * height_km
    
    st.write(f"**Selected Area:** `{round(area_sqkm, 2)} km²`")
    
    # --- 🚦 THE GATEKEEPER ---
    if area_sqkm < 0.1:
        st.error("❌ **Box is too small!** The satellite resolution is 10m/pixel. Please draw a larger box (at least 0.5 km²).")
        
    elif area_sqkm > 50.0:
        st.error(f"❌ **Box is too big!** You selected {round(area_sqkm, 2)} km². Please draw a box smaller than 50 km².")
        
    else:
        # The Goldilocks Zone! Show the success message AND the Run button
        st.success(f"✅ Target Area Locked! BBox: `{bbox}`")
        
        # 🚨 THE FIX: We use the new _obj variables instead of len(selected_dates)
        if start_date_obj <= end_date_obj:
            start_date = start_date_obj.strftime("%Y-%m-%d")
            end_date = end_date_obj.strftime("%Y-%m-%d")

            if st.button("🚀 Run U-Net Analysis", type="primary"):
                with st.spinner(f"Scanning satellite data from {start_date} to {end_date}..."):
                    try:
                        api_url = "http://127.0.0.1:8000/api/predict"
                        
                        payload = {
                            "bbox": bbox,
                            "start_date": start_date,
                            "end_date": end_date
                        }
                        response = requests.post(api_url, json=payload)
                        
                        if response.status_code == 200:
                            data = response.json()
                            
                            st.markdown("---")
                            st.subheader("📊 Analytics Report")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Capture Date", data["capture_date"])
                            col2.metric("Total Scan Area", f"{data['total_area_km^2']} km²")
                            col3.metric("Water Area", f"{data['water_area_km^2']} km²")
                            col4.metric("Water Coverage", f"{data['water_percentage']}%")
                            
                            img_bytes = base64.b64decode(data["mask_base64"])
                            mask_array = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
                            
                            # ... (Your existing code above this stays the same) ...
                            blue_mask = np.zeros((128, 128, 3), dtype=np.uint8)
                            blue_mask[mask_array == 255] = [0, 150, 255] 
                            
                            st.image(blue_mask, caption="AI Predicted Water Mask", width=400)
                            
                            # 🚨 PASTE THE NEW HISTORY SECTION HERE 🚨
                            st.markdown("---")
                            st.subheader("🗄️ Global Analysis History")
                            
                            with st.expander("📜 View Past Analyses (Fetched from PostgreSQL)"):
                                st.markdown("Fetching the latest satellite records from the database...")
                                try:
                                    # This calls the new endpoint we will build in FastAPI
                                    history_response = requests.get("http://127.0.0.1:8000/api/history")
                                    
                                    if history_response.status_code == 200:
                                        history_data = history_response.json()
                                        
                                        if len(history_data) > 0:
                                            # Convert the JSON into a beautiful Pandas table
                                            df = pd.DataFrame(history_data)
                                            
                                            # Reorder and rename columns for a professional UI (matching your exact keys)
                                            df = df[['capture_date', 'total_area_km^2', 'water_area_km^2', 'water_percentage']]
                                            df.columns = ['Capture Date', 'Total Area (km²)', 'Water Area (km²)', 'Water (%)']
                                            
                                            st.dataframe(df, use_container_width=True, hide_index=True)
                                        else:
                                            st.info("The database is currently empty. Run more analyses to build history!")
                                    else:
                                        st.warning("Could not fetch history. Is the database connected?")
                                        
                                except Exception as e:
                                    st.error(f"Database connection error: {e}")
                            # 🚨 END OF NEW SECTION 🚨

                        else:
                            st.error(f"Backend Error: {response.text}")
                        
                    except Exception as e:
                        st.error(f"Could not connect to FastAPI. Is your server running? Error: {e}")