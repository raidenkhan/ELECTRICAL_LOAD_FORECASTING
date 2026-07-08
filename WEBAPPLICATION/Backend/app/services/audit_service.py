import hashlib
import json
from datetime import datetime
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from app.db.models.audit_log import AuditLog
from app.core.logging import get_logger

logger = get_logger(__name__)


class AuditService:

    async def get_previous_hash(self, db: AsyncSession) -> str:
        result = await db.execute(
            select(AuditLog.hash)
            .order_by(desc(AuditLog.id))
            .limit(1)
        )
        row = result.scalar_one_or_none()
        return row or "0" * 64

    def _compute_hash(self, previous_hash: str, schedule_id: int, action: str, description: str, details: Optional[dict], user_id: Optional[int], timestamp: str) -> str:
        raw = f"{previous_hash}|{schedule_id}|{action}|{description}|{json.dumps(details or {}, sort_keys=True)}|{user_id}|{timestamp}"
        return hashlib.sha256(raw.encode()).hexdigest()

    async def log(
        self,
        db: AsyncSession,
        schedule_id: int,
        action: str,
        description: str,
        details: Optional[dict] = None,
        user_id: Optional[int] = None,
    ) -> AuditLog:
        previous_hash = await self.get_previous_hash(db)
        now = datetime.utcnow()
        timestamp = now.isoformat()
        hash_value = self._compute_hash(previous_hash, schedule_id, action, description, details, user_id, timestamp)

        entry = AuditLog(
            schedule_id=schedule_id,
            action=action,
            description=description,
            details=details,
            user_id=user_id,
            created_at=now,
            hash=hash_value,
            previous_hash=previous_hash,
        )
        db.add(entry)
        await db.flush()
        logger.debug(f"Audit log: [{action}] {description}")
        return entry

    async def get_for_schedule(self, db: AsyncSession, schedule_id: int) -> list[AuditLog]:
        result = await db.execute(
            select(AuditLog)
            .where(AuditLog.schedule_id == schedule_id)
            .order_by(AuditLog.id.asc())
        )
        return list(result.scalars().all())

    async def verify_chain(self, db: AsyncSession, schedule_id: int) -> dict:
        entries = await self.get_for_schedule(db, schedule_id)
        if not entries:
            return {"valid": True, "count": 0, "message": "No audit entries"}

        failures = []
        for i, entry in enumerate(entries):
            expected_hash = self._compute_hash(
                entry.previous_hash,
                entry.schedule_id,
                entry.action,
                entry.description,
                entry.details,
                entry.user_id,
                entry.created_at.isoformat() if entry.created_at else "",
            )
            if expected_hash != entry.hash:
                failures.append({"id": entry.id, "expected": expected_hash, "actual": entry.hash})

            if i > 0:
                if entry.previous_hash != entries[i - 1].hash:
                    failures.append({"id": entry.id, "chain_break": True, "expected_prev": entries[i - 1].hash, "actual_prev": entry.previous_hash})

        return {
            "valid": len(failures) == 0,
            "count": len(entries),
            "failures": failures,
        }
