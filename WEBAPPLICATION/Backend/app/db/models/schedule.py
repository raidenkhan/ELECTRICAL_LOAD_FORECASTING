from datetime import datetime, date
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Date, Text

from app.db.base import Base


class DailyDispatchSchedule(Base):
    __tablename__ = "daily_dispatch_schedules"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, nullable=False, unique=True, index=True)
    status = Column(String(50), default="draft")
    source_filename = Column(String(255), nullable=False)
    operator_notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class HourlyDemand(Base):
    __tablename__ = "hourly_demand"

    id = Column(Integer, primary_key=True, index=True)
    schedule_id = Column(Integer, nullable=False, index=True)
    hour = Column(Integer, nullable=False)
    entity_name = Column(String(100), nullable=False)
    demand_mw = Column(Float, nullable=False)
    is_forecasted = Column(Boolean, default=False)


class HourlySupply(Base):
    __tablename__ = "hourly_supply"

    id = Column(Integer, primary_key=True, index=True)
    schedule_id = Column(Integer, nullable=False, index=True)
    hour = Column(Integer, nullable=False)
    plant_name = Column(String(100), nullable=False)
    supply_mw = Column(Float, nullable=False)
