from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime
from app.db.base import Base


class BaseloadPlant(Base):
    __tablename__ = "baseload_plants"

    id = Column(Integer, primary_key=True, index=True)
    plant_name = Column(String(100), nullable=False)
    unit_name = Column(String(100), nullable=True)
    constant_mw = Column(Float, nullable=False)
    category = Column(String(50), nullable=False, default="thermal")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
