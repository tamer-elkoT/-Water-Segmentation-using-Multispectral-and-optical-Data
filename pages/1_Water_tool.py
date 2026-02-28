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

st.set_page_config(page_title="AI Engine", page_icon="🌍", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# 🧭 SIDEBAR
# ==========================================
with st.sidebar:
    st.title("AI Water Sentinel")
    st.divider()
    
    st.header("📅 Analysis Period")
    today = datetime.date.today()
    ninety_days_ago = today - datetime.timedelta(days=90)
    
    col_start, col_end = st.columns(2)
    with col_start:
        start_date_obj = st.date_input("Start Date", value=ninety_days_ago, max_value=today)
    with col_end:
        end_date_obj = st.date_input("End Date", value=today, max_value=today)
        
    st.divider()
    
    st.markdown("### 📬 Support & Links")
    st.markdown("[📧 Email Support](mailto:tamer.elkot.ai@gmail.com)")
    st.markdown("[💼 LinkedIn](https://www.linkedin.com/in/tamer-elkot/)")
    st.markdown("[🐙 GitHub Repository](https://github.com/tamer-elkoT)")
    st.markdown("[🌐 Portfolio](https://tamerelkot.netlify.app/)")

# ==========================================
# 🗂️ MAIN MAP TOOL
# ==========================================
st.markdown("### 🎯 Step 1: Target a Water Body")

locations = {
    "Aswan Low Dam, Egypt": [24.0350, 32.8750],
    "Nile Delta, Egypt": [31.4000, 31.0000],
    "Lake Nasser, Egypt": [22.7500, 32.7500],
    "Lake Victoria, Africa": [-1.0000, 33.0000],
    "Lake Mead, USA": [36.0163, -114.7370]
}
selected_loc = st.selectbox("🔍 Quick Jump to Location:", list(locations.keys()))
map_center = locations[selected_loc]

m = folium.Map(location=map_center, zoom_start=12, tiles="OpenStreetMap")
folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr='Esri', name='Satellite View'
).add_to(m)
folium.TileLayer(tiles='CartoDB positron', name='Minimalist (AI View)').add_to(m)
folium.LayerControl().add_to(m)
Geocoder().add_to(m)
LocateControl().add_to(m)
Draw(
    draw_options={'polyline': False, 'polygon': False, 'circle': False, 'marker': False, 'circlemarker': False, 'rectangle': True},
    edit_options={'edit': False}
).add_to(m)

map_output = st_folium(m, width="100%", height=500)

st.markdown("---")
st.markdown("### 🚀 Step 2: Run AI Extraction")

if map_output and map_output.get("last_active_drawing"):
    coordinates = map_output["last_active_drawing"]["geometry"]["coordinates"][0]
    lons = [p[0] for p in coordinates]
    lats = [p[1] for p in coordinates]
    bbox = [min(lons), min(lats), max(lons), max(lats)]
    
    avg_lat_rad = math.radians((bbox[1] + bbox[3]) / 2.0)
    height_km = (bbox[3] - bbox[1]) * 111.32
    width_km = (bbox[2] - bbox[0]) * 111.32 * math.cos(avg_lat_rad)
    area_sqkm = width_km * height_km
    
    st.write(f"**Selected Area:** `{round(area_sqkm, 2)} km²`")
    
    if area_sqkm < 0.1:
        st.error("❌ **Box is too small!** Please draw a larger box (at least 0.5 km²).")
    elif area_sqkm > 50.0:
        st.error(f"❌ **Box is too big!** You selected {round(area_sqkm, 2)} km².")
    else:
        st.success(f"✅ Target Area Locked! BBox: `{bbox}`")
        
        if start_date_obj <= end_date_obj:
            start_date = start_date_obj.strftime("%Y-%m-%d")
            end_date = end_date_obj.strftime("%Y-%m-%d")

            if st.button("⚙️ Execute U-Net", type="primary", use_container_width=True):
                with st.spinner(f"Downloading Sentinel-2 data from {start_date} to {end_date}..."):
                    try:
                        api_url = "http://127.0.0.1:8000/api/predict"
                        response = requests.post(api_url, json={"bbox": bbox, "start_date": start_date, "end_date": end_date})
                        
                        if response.status_code == 200:
                            data = response.json()
                            
                            st.markdown("---")
                            st.subheader("📊 Analytics Report")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Capture Date", data["capture_date"])
                            col2.metric("Total Scan Area", f"{data['total_area_sqkm']} km²")
                            col3.metric("Water Area", f"{data['water_area_sqkm']} km²")
                            col4.metric("Water Coverage", f"{data['water_percentage']}%")
                            
                            img_bytes = base64.b64decode(data["mask_base64"])
                            mask_array = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
                            
                            rgba_mask = np.zeros((128, 128, 4), dtype=np.uint8)
                            rgba_mask[mask_array == 255] = [0, 150, 255, 150] 
                            rgba_mask[mask_array == 0] = [0, 0, 0, 0]         
                            
                            _, buffer = cv2.imencode('.png', rgba_mask)
                            transparent_b64 = base64.b64encode(buffer).decode('utf-8')
                            image_uri = f"data:image/png;base64,{transparent_b64}"
                            
                            st.markdown("### 🗺️ Precision AI Map Overlay")
                            avg_lat = (bbox[1] + bbox[3]) / 2.0
                            avg_lon = (bbox[0] + bbox[2]) / 2.0
                            
                            result_map = folium.Map(
                                location=[avg_lat, avg_lon], zoom_start=14, 
                                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", attr="Esri"
                            )
                            
                            bounds = [[bbox[1], bbox[0]], [bbox[3], bbox[2]]] 
                            folium.raster_layers.ImageOverlay(
                                image=image_uri, bounds=bounds, name="AI Water Mask", opacity=1.0
                            ).add_to(result_map)
                            
                            st_folium(result_map, width="100%", height=500, key="final_map")
                        else:
                            st.error(f"Backend Error: {response.text}")
                    except Exception as e:
                        st.error(f"Server Error: {e}") 
        else:
            st.error("❌ The Start Date must be before the End Date.")
else:
    st.info("👈 Please draw a rectangle on the map to unlock the AI tool.", icon="ℹ️")