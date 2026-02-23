from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import setup_logging, get_logger
# Router registration
from app.api.v1.router import api_router

# Heartbeat to trigger reload: 2026-02-05 11:18
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    # Startup
    logger.info("Starting Load Forecasting API...")
    setup_logging()
    logger.info(f"API running in {'DEBUG' if settings.DEBUG else 'PRODUCTION'} mode")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Load Forecasting API...")


# Create FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Production-ready REST API for electrical load forecasting with STLF and LTLF capabilities",
    version="1.0.0",
    openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API v1 router
app.include_router(api_router, prefix=settings.API_V1_PREFIX)


@app.get("/", tags=["root"])
async def root():
    """Root endpoint - API information."""
    return {
        "name": settings.PROJECT_NAME,
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "service": "load-forecasting-api"
    }
