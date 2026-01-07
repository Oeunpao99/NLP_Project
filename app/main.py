"""
FastAPI application for Khmer NER
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager
import logging
import time
import os

from app.api.endpoints import router
from app.models.database import init_db
from app.services.ner_service import model_loader
from app.core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager"""
    # Startup
    logger.info("Starting up Khmer NER API...")
    
    try:
        # Initialize database
        init_db()
        logger.info("Database initialized")
        
        # Load ML model
        model_loader.load_model()
        logger.info("NER model loaded")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Khmer NER API...")

# Create FastAPI app
app = FastAPI(
    title="Khmer NER API",
    description="Named Entity Recognition for Khmer Language",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(router, prefix="/api/v1", tags=["ner"])

# Authentication routes
from app.api.auth import router as auth_router
app.include_router(auth_router, prefix="/api/v1", tags=["auth"])

# IMPORTANT: Mount static files BEFORE templates
# Mount CSS, JS, images
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Templates for HTML pages
templates = Jinja2Templates(directory="frontend")

# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    
    logger.info(
        f"{request.method} {request.url.path} "
        f"Status: {response.status_code} "
        f"Duration: {process_time:.2f}ms"
    )
    
    response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
    return response

@app.get("/")
async def root(request: Request):
    """Serve main page"""
    try:
        # Check if frontend directory exists
        if os.path.exists("frontend") and os.path.exists("frontend/index.html"):
            return templates.TemplateResponse(
                "index.html", 
                {"request": request}
            )
    except Exception as e:
        logger.error(f"Error serving frontend: {e}")
    
    # Fallback
    return {"message": "Khmer NER System", "frontend_status": "not_found"}

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return {}
@app.get("/history")
async def history_page(request: Request):
    """Serve history page"""
    return templates.TemplateResponse(
        "history.html", 
        {"request": request}
    )
@app.get("/profile")
async def profile_page(request: Request):
    """Serve the profile page"""
    return templates.TemplateResponse(
        "profile.html", 
        {"request": request}
    )