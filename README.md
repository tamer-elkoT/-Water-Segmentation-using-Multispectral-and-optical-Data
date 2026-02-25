# 🌍 Multispectral Satellite Water Segmentation 

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)
![Postgres](https://img.shields.io/badge/postgres-%23316192.svg?style=for-the-badge&logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)

An end-to-end deep learning pipeline that segments water bodies from 12-channel Sentinel-2 satellite imagery using a custom U-Net architecture. 

**Author:** Tamer | Full Stack AI Engineer  

---

## 📌 Project Overview
Traditional computer vision models rely on 3-channel (RGB) imagery. This project bypasses standard visual limitations by utilizing **12-channel multispectral arrays** (including Near-Infrared and Shortwave Infrared). By processing these physical spectral signatures, the neural network achieves high-precision water segmentation, resisting common visual false positives like dark shadows or black asphalt.

### Core Technologies:
* **Framework:** TensorFlow / Keras
* **Architecture:** 4-Level U-Net with Spatial Dropout and L2 Regularization
* **Data Engineering:** `tf.data.Dataset` API with custom Python bridges (`tifffile`, `cv2`)
* **Evaluation:** Custom Intersection over Union (IoU) metric and Binary Crossentropy

---

## 🏗️ Model Architecture & Pipeline
1. **Data Ingestion:** Raw `.tif` arrays are loaded and processed using Min-Max scaling to map physical reflectance values smoothly into a `[0.0, 1.0]` continuous float range.
2. **Augmentation Engine:** To preserve the mathematical integrity of the spectral bands, only strict geometric augmentations (random flips and 90-degree rotations) are applied on the fly during training.
3. **U-Net:** A contracting encoder extracts deep semantic features, while an expanding decoder uses skip connections to retain precise spatial boundaries. The output is a single-channel Sigmoid probability map.

**Visualize the Model Architecture locally:**
```bash
netron models/unet_water_seg_v1.h5
```
## 🗺️ User Scenario & Architecture Flow
The application features a fully interactive map. Users can draw a bounding box over any area on Earth, and the backend dynamically fetches the live 12-channel satellite data, runs the U-Net inference, logs the statistics, and returns the highlighted water mask.

(Ensure the Mermaid SVG is exported and saved as user_scenario.svg in an assets/ folder)
```
# 🗂️ Project Structure (MVC Architecture)
water_segmentation_project/
│
├── ml_pipeline/                   # 🧪 OFFLINE: Where the model was trained
│   ├── data/                      # Raw and preprocessed .tif files
│   ├── notebooks/                 # Jupyter exploratory files
│   ├── src/                       # Training scripts (unet_model.py, augment.py)
│   └── weights/                   # unet_water_seg_v1.h5 (Trained weights)
│
├── app/                           # 🚀 ONLINE: The Live MVC Application
│   ├── main.py                    # The application engine (Starts FastAPI & Streamlit)
│   │
│   ├── models/                    # 🧠 MODEL: Data & Intelligence Layer
│   │   ├── __init__.py
│   │   ├── database.py            # SQLAlchemy tables (e.g., PredictionLog)
│   │   └── unet_inference.py      # Loads .h5 model and runs model.predict()
│   │
│   ├── views/                     # 🖥️ VIEW: User Interface Layer
│   │   ├── __init__.py
│   │   └── streamlit_ui.py        # The interactive map and dashboard
│   │
│   ├── controllers/               # 🚦 CONTROLLER: Business Logic & Routing
│   │   ├── __init__.py
│   │   ├── api_router.py          # FastAPI endpoints (e.g., /api/predict)
│   │   └── satellite_service.py   # PySTAC logic to download data via coordinates
│   │
│   └── core/                      # ⚙️ Configuration
│       ├── config.py              # Environment variables (DB passwords, API keys)
│       └── db_setup.py            # PostgreSQL connection pool
│
├── assets/                        # Images and diagrams for documentation
├── requirements.txt               # Dependencies
├── Dockerfile                     # Instructions to build the web app container
├── docker-compose.yml             # Spins up the App and PostgreSQL database
└── README.md
```
# 🛤️ Development Phases
This project is being built in 4 distinct phases:

[x] Phase 1: The ML Engine: Train the 12-channel U-Net and write the PySTAC script to fetch live satellite arrays via GPS coordinates.

[ ] Phase 2: The Backend API: Wrap the inference engine in FastAPI to handle asynchronous web requests.

[ ] Phase 3: The Interactive Frontend: Build the Streamlit UI with folium integration to allow users to draw bounding boxes on a live map.

[ ] Phase 4: Database & Deployment: Implement PostgreSQL logging via SQLAlchemy and Dockerize the entire stack for cloud deployment.

🚀 Setup and Installation
(This section will be expanded as Docker and Database integration is completed)

1. Clone the repository
```bash
git clone [https://github.com/tamer-elkoT/Water-Segmentation-using-Multispectral-and-optical-Data.git]
cd water_segmentation_project
```
2. Create a virtual environment
```bash
conda create -n water_seg python=3.10
conda activate water_seg
```
3. Install dependencies
```bash
pip install -r requirements.txt
```

![End-to-End User Scenario](/mnt/d/01_Projects/CV/Water_Segmentation/assets/User_Senario.svg)

Are we ready to dive into **Phase 2** and start writing the `api_router.py` (FastAPI endpoints) to connect your data fetcher with your U-Net model?