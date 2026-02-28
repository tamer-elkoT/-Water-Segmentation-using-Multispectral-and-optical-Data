# import streamlit as st
# import folium
# from folium.plugins import Draw, Geocoder, LocateControl
# from streamlit_folium import st_folium
# import requests
# import base64
# import numpy as np
# import cv2
# import math
# import datetime

# # 1. Page Configuration
# st.set_page_config(page_title="AI Water Sentinel", page_icon="🌍", layout="wide", initial_sidebar_state="expanded")

# # 🚨 THE ROUTER MEMORY: Default is now 'home_page'
# if 'current_page' not in st.session_state:
#     st.session_state.current_page = 'home_page'

# def switch_page(page_name):
#     st.session_state.current_page = page_name

# # ==========================================
# # 🧭 SIDEBAR: CONTROLS & NAVIGATION
# # ==========================================
# with st.sidebar:
#     st.title("AI Water Sentinel")
#     st.markdown("Multispectral Satellite Intelligence")
#     st.divider()
    
#     # Show the "Back to Home" button ONLY if they are inside the tool
#     if st.session_state.current_page == 'tool_page':
#         st.button("🏠 Back to Home", on_click=switch_page, args=('home_page',), use_container_width=True)
#         st.divider()
    
#     st.header("📅 Analysis Period")
#     today = datetime.date.today()
#     ninety_days_ago = today - datetime.timedelta(days=90)
    
#     col_start, col_end = st.columns(2)
#     with col_start:
#         start_date_obj = st.date_input("Start Date", value=ninety_days_ago, max_value=today)
#     with col_end:
#         end_date_obj = st.date_input("End Date", value=today, max_value=today)
        
#     st.divider()
    
#     st.header("👨‍💻 About the Author")
#     st.markdown("**Tamer** *AI Engineer* 📍 Cairo, Egypt")
    
#     st.markdown("### 📬 Support & Links")
#     st.markdown("[📧 Email Support](tamer.elkot.ai@gmail.com)")
#     st.markdown("[💼 LinkedIn](https://www.linkedin.com/in/tamer-elkot/)")
#     st.markdown("[🐙 GitHub Repository](https://github.com/tamer-elkoT)")
#     st.markdown("[🌐 Portfolio](https://tamerelkot.netlify.app/)")


# # ==========================================
# # 🗂️ VIEW 1: THE HOME PAGE (Default)
# # ==========================================
# if st.session_state.current_page == 'home_page':
#     st.title("🌊 AI Water Sentinel: Global Water Monitoring")
#     st.markdown("### Empowering organizations with Deep Learning and Multispectral Satellite Intelligence.")
#     st.divider()
    
#     col1, col2 = st.columns(2)
#     with col1:
#         st.header("🌍 The Global Impact")
#         st.markdown("""
#         Water is our most critical resource. Climate change, urban expansion, and agricultural demands are rapidly altering the Earth's surface water.
        
#         **AI Water Sentinel** leverages the European Space Agency's Sentinel-2 satellite network and a custom **U-Net Neural Network** to track these changes with mathematical precision. 
#         """)
#     with col2:
#         st.header("🎯 Target Audience")
#         st.markdown("""
#         * **🏛️ Government & Urban Planners**
#         * **🌾 Agricultural Ministries**
#         * **🚁 Disaster Response Teams**
#         * **🔬 Climate Scientists**
#         """)
    
#     st.divider()
    
#     # 🚨 THE CALL TO ACTION BUTTON (Right in the middle!)
#     st.markdown("<h2 style='text-align: center;'>Ready to analyze live satellite data?</h2>", unsafe_allow_html=True)
    
#     # We use columns to center the button perfectly
#     _, center_col, _ = st.columns([1, 2, 1])
#     with center_col:
#         st.button("🚀 Try It Out (Launch AI Engine)", on_click=switch_page, args=('tool_page',), use_container_width=True, type="primary")
    
#     st.divider()
    
#     st.markdown("<div style='text-align: center; color: gray;'>", unsafe_allow_html=True)
#     st.markdown("Built with Python, FastAPI, TensorFlow, and Microsoft Planetary Computer")
#     st.markdown("</div>", unsafe_allow_html=True)


# # ==========================================
# # 🗂️ VIEW 2: THE AI TOOL
# # ==========================================
# elif st.session_state.current_page == 'tool_page':
#     st.markdown("### 🎯 Step 1: Target a Water Body")
    
#     locations = {
#         "Aswan Low Dam, Egypt": [24.0350, 32.8750],
#         "Nile Delta, Egypt": [31.4000, 31.0000],
#         "Lake Nasser, Egypt": [22.7500, 32.7500],
#         "Lake Victoria, Africa": [-1.0000, 33.0000],
#         "Lake Mead, USA": [36.0163, -114.7370]
#     }
#     selected_loc = st.selectbox("🔍 Quick Jump to Location:", list(locations.keys()))
#     map_center = locations[selected_loc]

#     # Initialize the Map
#     m = folium.Map(location=map_center, zoom_start=12, tiles="OpenStreetMap")
#     folium.TileLayer(
#         tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
#         attr='Esri', name='Satellite View'
#     ).add_to(m)
#     folium.TileLayer(tiles='CartoDB positron', name='Minimalist (AI View)').add_to(m)
#     folium.LayerControl().add_to(m)
#     Geocoder().add_to(m)
#     LocateControl().add_to(m)
#     Draw(
#         draw_options={'polyline': False, 'polygon': False, 'circle': False, 'marker': False, 'circlemarker': False, 'rectangle': True},
#         edit_options={'edit': False}
#     ).add_to(m)

#     # Display Map
#     map_output = st_folium(m, width="100%", height=500)

#     st.markdown("---")
#     st.markdown("### 🚀 Step 2: Run AI Extraction")

#     # The AI Execution Logic
#     if map_output and map_output.get("last_active_drawing"):
#         coordinates = map_output["last_active_drawing"]["geometry"]["coordinates"][0]
#         lons = [p[0] for p in coordinates]
#         lats = [p[1] for p in coordinates]
#         bbox = [min(lons), min(lats), max(lons), max(lats)]
        
#         avg_lat_rad = math.radians((bbox[1] + bbox[3]) / 2.0)
#         height_km = (bbox[3] - bbox[1]) * 111.32
#         width_km = (bbox[2] - bbox[0]) * 111.32 * math.cos(avg_lat_rad)
#         area_sqkm = width_km * height_km
        
#         st.write(f"**Selected Area:** `{round(area_sqkm, 2)} km²`")
        
#         if area_sqkm < 0.1:
#             st.error("❌ **Box is too small!** Please draw a larger box (at least 0.5 km²).")
#         elif area_sqkm > 50.0:
#             st.error(f"❌ **Box is too big!** You selected {round(area_sqkm, 2)} km². Please draw a box smaller than 50 km².")
#         else:
#             st.success(f"✅ Target Area Locked! BBox: `{bbox}`")
            
#             if start_date_obj <= end_date_obj:
#                 start_date = start_date_obj.strftime("%Y-%m-%d")
#                 end_date = end_date_obj.strftime("%Y-%m-%d")

#                 if st.button("⚙️ Execute U-Net", type="primary", use_container_width=True):
#                     with st.spinner(f"Downloading Sentinel-2 data from {start_date} to {end_date}..."):
#                         try:
#                             api_url = "http://127.0.0.1:8000/api/predict"
#                             response = requests.post(api_url, json={"bbox": bbox, "start_date": start_date, "end_date": end_date})
                            
#                             if response.status_code == 200:
#                                 data = response.json()
                                
#                                 st.markdown("---")
#                                 st.subheader("📊 Analytics Report")
                                
#                                 col1, col2, col3, col4 = st.columns(4)
#                                 col1.metric("Capture Date", data["capture_date"])
#                                 col2.metric("Total Scan Area", f"{data['total_area_km^2']} km²")
#                                 col3.metric("Water Area", f"{data['water_area_km^2']} km²")
#                                 col4.metric("Water Coverage", f"{data['water_percentage']}%")
                                
#                                 # Unpack and display the image overlay
#                                 img_bytes = base64.b64decode(data["mask_base64"])
#                                 mask_array = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
                                
#                                 rgba_mask = np.zeros((128, 128, 4), dtype=np.uint8)
#                                 rgba_mask[mask_array == 255] = [0, 150, 255, 150] 
#                                 rgba_mask[mask_array == 0] = [0, 0, 0, 0]         
                                
#                                 _, buffer = cv2.imencode('.png', rgba_mask)
#                                 transparent_b64 = base64.b64encode(buffer).decode('utf-8')
#                                 image_uri = f"data:image/png;base64,{transparent_b64}"
                                
#                                 st.markdown("### 🗺️ Precision AI Map Overlay")
#                                 avg_lat = (bbox[1] + bbox[3]) / 2.0
#                                 avg_lon = (bbox[0] + bbox[2]) / 2.0
                                
#                                 result_map = folium.Map(
#                                     location=[avg_lat, avg_lon], zoom_start=14, 
#                                     tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", attr="Esri"
#                                 )
                                
#                                 bounds = [[bbox[1], bbox[0]], [bbox[3], bbox[2]]] 
#                                 folium.raster_layers.ImageOverlay(
#                                     image=image_uri, bounds=bounds, name="AI Water Mask", opacity=1.0
#                                 ).add_to(result_map)
                                
#                                 st_folium(result_map, width="100%", height=500, key="final_map")
#                             else:
#                                 st.error(f"Backend Error: {response.text}")
#                         except Exception as e:
#                             st.error(f"Server Error: {e}") 
#             else:
#                 st.error("❌ The Start Date must be before the End Date.")
#     else:
#         st.info("👈 Please draw a rectangle on the map to unlock the AI tool.", icon="ℹ️")


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
                            
                            blue_mask = np.zeros((128, 128, 3), dtype=np.uint8)
                            blue_mask[mask_array == 255] = [0, 150, 255] 
                            
                            st.image(blue_mask, caption="AI Predicted Water Mask", width=400)
                            
                        else:
                            st.error(f"Backend Error: {response.text}")
                            
                    except Exception as e:
                        st.error(f"Could not connect to FastAPI. Is your server running? Error: {e}") 
                        
        else:
            # This triggers if they pick a Start Date that is AFTER the End Date
            st.error("❌ **Time Travel Detected!** The Start Date must be before the End Date.")

# This runs if the user hasn't drawn a box yet
else:
    st.info("👈 Please draw a rectangle on the map to begin.", icon="ℹ️")



