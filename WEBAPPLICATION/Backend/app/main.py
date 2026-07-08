import os
import stat as stat_module
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException

from app.core.config import settings
from app.core.logging import setup_logging, get_logger
# Router registration
from app.api.v1.router import api_router
# Startup seed
from app.db.session import AsyncSessionLocal
from app.services.baseload_service import BaseloadService
from app.ml.auto_monitor import start_monitor, stop_monitor

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

    # Seed baseload plants on first run
    try:
        async with AsyncSessionLocal() as db:
            service = BaseloadService()
            count = await service.seed_if_empty(db)
            if count:
                logger.info(f"Seeded {count} baseload plants into database")
    except Exception as e:
        logger.warning(f"Baseload seeding skipped: {e}")

    logger.info(f"API running in {'DEBUG' if settings.DEBUG else 'PRODUCTION'} mode")

    # Start auto drift monitor
    monitor_task = await start_monitor()
    logger.info(f"AutoMonitor started (task_id={id(monitor_task)})")

    yield

    # Shutdown: stop monitor
    await stop_monitor(monitor_task)
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

# Serve static frontend (built by `npm run build` in frontend/)
_frontend_dir = settings.FRONTEND_DIR
if _frontend_dir.is_dir():
    class _NextStaticFiles(StaticFiles):
        async def get_response(self, path: str, scope):
            full_path, stat_result = self.lookup_path(path)
            if stat_result is not None and stat_module.S_ISDIR(stat_result.st_mode):
                html_path = path + ".html"
                html_full, html_stat = self.lookup_path(html_path)
                if html_stat is not None and not stat_module.S_ISDIR(html_stat.st_mode):
                    return await super().get_response(html_path, scope)
            return await super().get_response(path, scope)

    app.mount("/", _NextStaticFiles(directory=str(_frontend_dir), html=True), name="frontend")
    logger.info(f"Serving frontend from {_frontend_dir}")
else:
    logger.info(f"No frontend build found at {_frontend_dir} â€” API-only mode")


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



