from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON
from app.db.base import Base


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    schedule_id = Column(Integer, ForeignKey("daily_dispatch_schedules.id"), nullable=False, index=True)
    action = Column(String(50), nullable=False)
    description = Column(Text, nullable=False)
    details = Column(JSON, nullable=True)
    user_id = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    hash = Column(String(64), nullable=False)
    previous_hash = Column(String(64), nullable=False, default="0" * 64)
