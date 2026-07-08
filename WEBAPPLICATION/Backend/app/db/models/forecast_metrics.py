from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Date, JSON
from app.db.base import Base


class ForecastMetrics(Base):
    __tablename__ = "forecast_metrics"

    id = Column(Integer, primary_key=True, index=True)
    forecast_date = Column(Date, nullable=False, index=True)
    horizon = Column(String(10), nullable=False)
    actual_mw = Column(JSON, nullable=True)
    forecast_mw = Column(JSON, nullable=False)
    mae = Column(Float, nullable=False)
    mape = Column(Float, nullable=True)
    engine = Column(String(50), default="dlinear_h10")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
