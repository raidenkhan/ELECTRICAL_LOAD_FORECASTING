from fastapi import APIRouter

from app.api.v1 import data, forecast, explain, models
from app.api.v1.endpoints import login, users

api_router = APIRouter()

# Include all v1 routers
api_router.include_router(data.router, prefix="/data", tags=["data"])
api_router.include_router(forecast.router, prefix="/forecast", tags=["forecast"])
api_router.include_router(explain.router, prefix="/explain", tags=["explain"])
api_router.include_router(models.router, prefix="/models", tags=["models"])
api_router.include_router(login.router, tags=["login"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
