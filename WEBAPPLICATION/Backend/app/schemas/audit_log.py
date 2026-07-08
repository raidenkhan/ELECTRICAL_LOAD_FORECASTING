from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class AuditLogEntry(BaseModel):
    id: int
    schedule_id: int
    action: str
    description: str
    details: Optional[dict] = None
    user_id: Optional[int] = None
    created_at: datetime
    hash: str
    previous_hash: str

    class Config:
        from_attributes = True


class ChainVerificationResult(BaseModel):
    valid: bool
    count: int
    failures: list[dict] = []
