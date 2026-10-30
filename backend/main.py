import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.services.cinematography_model import load_cinematography_models
from backend.routes.analysis import router as analysis_router
from backend.routes.auth import router as auth_router

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("cinematography_backend")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan context manager to load CLIP and joblib models ONCE at application startup.
    """
    logger.info("==================================================")
    logger.info("STARTING CINEMATOGRAPHY EVALUATION BACKEND ENGINE")
    logger.info("==================================================")
    try:
        load_cinematography_models()
        logger.info("All 5 CineTechBench classifiers and CLIP loaded into memory.")
    except Exception as e:
        logger.error(f"Failed to load models during startup: {e}")
    yield
    logger.info("Shutting down Cinematography Evaluation Backend.")

app = FastAPI(
    title="CineTechBench Cinematography Evaluation API",
    description="FastAPI backend integrating CLIP, OpenCV Optical Flow, and CineTechBench classifiers for film intent scoring.",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routes
app.include_router(analysis_router)
app.include_router(auth_router)

# Mount workspace root for static html files (main.html, profile.html, login.html, etc.)
workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app.mount("/", StaticFiles(directory=workspace_root, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
