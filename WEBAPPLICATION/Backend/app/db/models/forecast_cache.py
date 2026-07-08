from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, JSON, Date
from app.db.base import Base


class ForecastCache(Base):
    __tablename__ = "forecast_cache"

    id = Column(Integer, primary_key=True, index=True)
    cache_key = Column(String(100), unique=True, nullable=False, index=True)
    horizon = Column(String(10), nullable=False)
    forecast_date = Column(Date, nullable=False)
    data = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
