from fastapi import FastAPI
from app.controllers.api_router import router

# intialize the fastapi app
app = FastAPI(
    title="🌍 Sentinel-2 Water Segmentation API",
    description="A high-performance backend for multispectral Earth Observation AI.",
    version="1.0.0"
)
# db_models.Base.metadata.create_all(bind=engine)
# 2. Mount the API Router
# All routes in api_router.py will now have the "/api" prefix (e.g., /api/predict)
app.include_router(router, prefix="/api", tags=["Inference"])

# 3. Create a simple health check endpoint
@app.get("/")
def health_check():
    """Simple endpoint to verify the server is running."""
    return {
        "status": "Online",
        "service": "Water Segmentation Backend",
        "ai_engine": "Loaded and Ready"
    }