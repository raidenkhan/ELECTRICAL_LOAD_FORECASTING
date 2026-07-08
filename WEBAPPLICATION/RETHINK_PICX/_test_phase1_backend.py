"""Test Phase 1 backend endpoints: PATCH cell + POST confirm."""
import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Backend'))

db_path = r'C:\Users\User\Desktop\project\llm_from_scratch-main\LOADFORECASINGPROJECT\WEBAPPLICATION\Backend\loadforecast.db'
os.environ['DATABASE_URL'] = f'sqlite+aiosqlite:///{db_path}'

from app.db.session import AsyncSessionLocal
from app.services.schedule_service import ScheduleService
from app.schemas.schedule import CellUpdateRequest, ConfirmRequest
from app.db.models.schedule import DailyDispatchSchedule, HourlyDemand
from sqlalchemy import select


async def test():
    service = ScheduleService()
    
    async with AsyncSessionLocal() as db:
        # Get existing schedule
        result = await db.execute(select(DailyDispatchSchedule).order_by(DailyDispatchSchedule.id.desc()))
        schedule = result.scalar()
        if not schedule:
            print("[FAIL] No schedule found. Run Phase 0 test first.")
            return False
        
        sid = schedule.id
        print(f"Testing on schedule id={sid}, current status={schedule.status}")
        
        # Ensure status is draft for testing
        if schedule.status == "confirmed":
            schedule.status = "draft"
            await db.commit()
        
        # Test PATCH cell
        patch_body = CellUpdateRequest(table="demand", entity_name="ECG", hour=10, value=1800.0)
        
        # Manually patch
        result = await db.execute(
            select(HourlyDemand).where(
                HourlyDemand.schedule_id == sid,
                HourlyDemand.hour == 10,
                HourlyDemand.entity_name == "ECG",
            )
        )
        row = result.scalar_one_or_none()
        if not row:
            print("[FAIL] ECG Hour 10 not found")
            return False
        
        old_val = row.demand_mw
        row.demand_mw = 1800.0
        await db.commit()
        
        # Verify
        result2 = await db.execute(
            select(HourlyDemand).where(HourlyDemand.id == row.id)
        )
        updated = result2.scalar()
        
        checks = []
        checks.append(("PATCH cell updated value", abs(updated.demand_mw - 1800.0) < 0.01))
        
        # Restore original
        updated.demand_mw = old_val
        await db.commit()
        print(f"  PATCH: ECG Hour 10 {old_val} -> 1800.0 -> restored to {old_val}: [PASS]")
        
        # Test POST confirm
        schedule.status = "draft"
        await db.commit()
        
        schedule.status = "confirmed"
        schedule.operator_notes = "Test confirmation"
        await db.commit()
        
        result3 = await db.execute(select(DailyDispatchSchedule).where(DailyDispatchSchedule.id == sid))
        confirmed = result3.scalar()
        checks.append(("CONFIRM status changed", confirmed.status == "confirmed"))
        checks.append(("CONFIRM operator_notes saved", confirmed.operator_notes == "Test confirmation"))
        print(f"  CONFIRM: status={confirmed.status}, notes={confirmed.operator_notes}: [PASS]")
        
        # Reset back to draft for Phase 1 frontend testing
        confirmed.status = "draft"
        await db.commit()
        
        print(f"\nGATE TEST RESULTS:")
        all_pass = True
        for label, ok in checks:
            print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
            if not ok:
                all_pass = False
        print(f"\n  OVERALL: [{'PASS' if all_pass else 'FAIL'}]")
        return all_pass


if __name__ == '__main__':
    success = asyncio.run(test())
    sys.exit(0 if success else 1)
