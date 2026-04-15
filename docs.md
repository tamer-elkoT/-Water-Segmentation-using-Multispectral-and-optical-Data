# 🌊 AI Water Sentinel: Multispectral Satellite Water Segmentation

An end-to-end cloud-native deep learning platform that precisely segments water bodies from live 12-channel Sentinel-2 satellite imagery using a custom U-Net architecture. By processing multispectral physical signatures instead of traditional visible light, the application achieves high-precision water segmentation on a global scale.

**Author:** Tamer | Full Stack AI Engineer  

---

## 📌 Project Overview Detailed

Standard computer vision networks traditionally rely on 3-channel (RGB) images, making them susceptible to visual misinterpretations where dark shadows, dense forests, or black asphalt are wrongly classified as water. 

This project solves this limitation by implementing a **12-channel multispectral AI engine**. It digests not just visible light, but Near-Infrared (NIR), Shortwave Infrared (SWIR), and mathematically synthesized geographic probability bands. Combining Deep Learning with a modern high-performance web architecture, the system allows users to interactively draw bounding boxes anywhere on Earth, fetching live satellite geospatial data, running real-time neural network inference, and rendering the highly precise binary water mask instantly.

This project is meticulously architected following the **MVC (Model-View-Controller)** pattern, ensuring separation of concerns, scalability, and robust deployment pipelines using Docker.

---

## 🛠️ Tools & Technologies

This ecosystem incorporates a broad range of cutting-edge technologies to bridge Machine Learning with full-stack web development.

### Machine Learning & Data Science
- **TensorFlow / Keras:** The primary deep learning framework for defining, training, and running inference on the custom U-Net neural network.
- **NumPy & SciPy:** Used extensively for advanced multidimensional array manipulation.
- **Scikit-learn:** Employed in metrics calculation and data pipeline structuring.

### Geospatial & Image Processing
- **Rasterio & CV2 (OpenCV):** Used for robust reading of cloud-optimized GeoTIFFs, cropping strict bounding boxes (`rasterio.windows`), and uniformly resizing (`cv2.resize`) the multispectral bands.
- **PySTAC-Client & Planetary Computer:** Used to query Microsoft's open satellite database STAC catalogs to locate and authenticate the latest cloud-free Sentinel-2 satellite tiles dynamically.
- **Folium:** An interactive JavaScript map rendered in Python, enabling geospatial user inputs and displaying the AI overlay maps.

### Backend APIs & Architecture (MVC)
- **FastAPI:** The high-performance asynchronous python web framework hosting the inference engine and database logic (Acts as the **Controller / API Router**).
- **Uvicorn:** The ASGI web server handling FastAPI requests.
- **SQLAlchemy & GeoAlchemy2:** The Object-Relational Mapper (ORM) powering the database schemas, enabling robust SQL insertion of prediction statistics (Acts as the **Model**).
- **PostgreSQL:** A scalable relational database system.

### Frontend User Interface
- **Streamlit:** A reactive Python-based web framework used to build our Data Dashboard and Map Interface (Acts as the **View**).

### Deployment
- **Docker & Docker Compose:** Used to containerize the FastAPI backend, Streamlit frontend, and spinning up an isolated PostgreSQL database.

---

## 🚦 Application Workflow & Data Flow

The platform relies on a seamless pipeline integrating the User Interface, API, Data Fetching, Inference Engine, and Database.

1. **User Input (`views/streamlit_ui.py`):**
   - The user opens the Streamlit web dashboard.
   - Using a **Folium** interactive map, they draw a geometric bounding box over a region on Earth and define a timeframe date range.
   - Streamlit instantly packages these GPS coordinates natively into a JSON payload and pushes a `POST` request to the backend.

2. **Backend Routing & Orchestration (`controllers/api_router.py`):**
   - **FastAPI** catches the payload at the `/api/predict` endpoint.
   - It calculates the total real-world Square Kilometers enclosed within the bounding box using rigorous geographic math.

3. **Data Ingestion (`controllers/satellite_service.py`):**
   - The backend triggers PySTAC to poll the Microsoft Planetary Computer. It finds the highest quality imagery within the user-specified timeframe (cloud cover < 5%).
   - Using highly optimized GDAL configurations (`rasterio.env`), it performs a partial download. Instead of downloading heavy multi-gigabyte files, it natively crops and downloads exclusively the pixels within the user's bounding box across optical and infrared channels.
   - Geometric probability bands (like NDWI, JRC Water Probability) are dynamically synthesized. All bands are resized via `cv2` to `128x128` and vertically stacked.
   - **Crucial Step:** The array undergoes strict Min-Max Normalization to map all physical reflectance values optimally between `(0.0, 1.0)`.

4. **Neural Engine Prediction (`models/unet_inference.py`):**
   - The normalized 12-channel `(128, 128, 12)` NumPy array is batched and passed into the `WaterSegmentationModel` class.
   - The U-Net executes its feed-forward pass, contracting features and expanding spatial mappings, returning a raw Sigmoid probability mask.
   - A `> 0.5` binary threshold executes, generating a strict array where `1` represents Water and `0` represents Land.

5. **Data Post-Processing & Persistence (`models/database.py` & `api_router.py`):**
   - The binary mask calculates total water area and coverage percentage. 
   - The array is multiplied by `255`, encoded entirely into a lossless Base64 PNG string (`cv2.imencode`).
   - The system triggers an SQLAlchemy Database session, permanently logging the bounding box, capture date, total area, and water percentage into PostgreSQL.

6. **Frontend Rendering:**
   - The FastAPI server responds with the Base64 image payload and calculated statistics.
   - Streamlit decodes the Base64 string, dynamically maps water pixels to an opaque translucent blue color map, and overlays it atop the original map view natively utilizing Folium's `ImageOverlay` anchored bounds.

---

## 🏗️ Project Architecture (MVC Pattern)

```text
water_segmentation_project/
│
├── ml_pipeline/                   # 🧪 OFFLINE: Where the model was trained
│   ├── data/                      # Raw and preprocessed .tif files
│   ├── notebooks/                 # Jupyter exploratory files
│   └── weights/                   # unet_water_seg_v1.h5 (Trained weights)
│
├── app/                           # 🚀 ONLINE: The Live MVC Application
│   ├── main.py                    # The core application engine (FastAPI Root)
│   │
│   ├── models/                    # 🧠 MODEL: Data & Intelligence Layer
│   │   ├── database.py            # SQLAlchemy tables (e.g., PredictionLog)
│   │   └── unet_inference.py      # Object-Oriented U-Net Loader & Predictor
│   │
│   ├── views/                     # 🖥️ VIEW: User Interface Layer
│   │   └── (Linked via pages/1_Water_tool.py)
│   │
│   ├── controllers/               # 🚦 CONTROLLER: Business Logic & Routing
│   │   ├── api_router.py          # FastAPI endpoint /api/predict
│   │   └── satellite_service.py   # PySTAC planetary connection logic
│   │
│   └── core/                      # ⚙️ SYSTEM: Configuration
│       ├── config.py              # Environment variables
│       └── db_setup.py            # PostgreSQL connection pool setup
│
├── pages/                         # Streamlit sub-pages
│   └── 1_Water_tool.py            # The Interactive AI Tool User Interface
│
├── requirements.txt               # Dependencies
├── Dockerfile                     # Instructions to build the web app container
└── docker-compose.yml             # Orchestrates the App and PostgreSQL database
```

---

## 🧗 Problems Faced & Technical Solutions

### Problem 1: Out of Memory (OOM) Errors & Latency Bottlenecks
**Issue:** Loading standard Keras `.h5` model files during every single `/api/predict` API call was incredibly slow (taking 3-5 seconds per request) and routinely crashed the web server when memory boundaries shattered on multiple requests.  
**Solution:** Adopted a Singleton Object-Oriented layout (`WaterSegmentationModel`). The U-Net weights are now loaded exclusively **once** into RAM/VRAM when the FastAPI server initializes. API calls now merely access the loaded model instance, cutting inference latency down to ~80ms.

### Problem 2: Multispectral Input Dimension Mismatch (Band Resolutions)
**Issue:** Standard Sentinel-2 optical bands (RGB/NIR) operate at 10m/pixel resolutions, while SWIR bands inherently operate at 20m/pixel. A U-Net rigidly expects fully symmetric spatial mappings and will crash if array depths mismatch.  
**Solution:** Engineered a targeted Open-CV (`cv2.resize`) bridge within the data loader. It loops through all varied geographic bands dynamically interpolating and forcing them into a strict homogeneous `128x128` spatial standard matrix before stacking the depth channels.

### Problem 3: Lethal Model Accuracy Drops (The "Black Mask" Incident)
**Issue:** Post-deployment, the U-Net began returning purely black masks (0 water detected) during inference, despite achieving 98% IoU during training.  
**Solution:** Upon severe debugging, it was discovered that live planetary APIs return unscaled 16-bit physical radiance values (ranging wildly from 0 to 15,000+), blinding the sigmoid functions. 
A rigid `Min-Max Scaler` equation `(array - min) / (max - min)` was instituted in `satellite_service.py`, harmonizing exactly with the original training pipeline and normalizing all runtime arrays strictly between `0.0` and `1.0`.

### Problem 4: Sluggish Gigabyte Cloud Downloads
**Issue:** Standard file fetching would forcibly pull entire `100km x 100km` satellite tiles taking 10+ minutes, vastly exceeding API timeout thresholds simply to analyze a tiny bounding box.  
**Solution:** Leveraged highly optimized Cloud-Optimized GeoTIFFs (COGs). Enabled hidden GDAL environment variables (`rasterio.env.Env(VSI_CACHE=True)`), meaning Rasterio solely downloads exclusively the granular bytes representing the exact pixel bounds requested over the network (Partial Range HTTP Requests).

### Problem 5: Moving massive Arrays between Backend and Frontend
**Issue:** Streaming a raw `numpy` matrix of inference results across HTTP from FastAPI to Streamlit was vastly inefficient and heavy.  
**Solution:** Applied an elegant byte-compression. The `(128, 128)` mask is inherently transformed by Open-CV directly into an optimized byte-buffer `.png` image dynamically within RAM. It is base64 encoded into a raw string, zipped within JSON, and reconstructed fully on the frontend. This practically eliminated networking payload burdens.

---

*This application pushes the boundaries of how geospatial AI systems are developed, shifting purely functional notebooks into fully operational, state-of-the-art interactive cloud platforms.*
