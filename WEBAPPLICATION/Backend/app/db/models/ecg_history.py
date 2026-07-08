from datetime import datetime, date
from sqlalchemy import Column, Integer, Float, DateTime, Date, Boolean

from app.db.base import Base


class EcgHistoricalDemand(Base):
    __tablename__ = "ecg_historical_demand"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, nullable=False, index=True)
    hour = Column(Integer, nullable=False)
    demand_mw = Column(Float, nullable=False)
    temperature_c = Column(Float, nullable=True)
    is_holiday = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
