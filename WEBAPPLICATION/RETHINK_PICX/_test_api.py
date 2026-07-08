"""Test the GET /schedule/{id} endpoint via the service layer (no server needed)."""
import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Backend'))

db_path = r'C:\Users\User\Desktop\project\llm_from_scratch-main\LOADFORECASINGPROJECT\WEBAPPLICATION\Backend\loadforecast.db'
os.environ['DATABASE_URL'] = f'sqlite+aiosqlite:///{db_path}'

from app.db.session import AsyncSessionLocal
from app.services.schedule_service import ScheduleService
from sqlalchemy import select
from app.db.models.schedule import DailyDispatchSchedule


async def test_get():
    service = ScheduleService()
    async with AsyncSessionLocal() as db:
        result = await service.get_schedule(db, schedule_id=1)
        
        if result is None:
            print("[FAIL] Schedule not found")
            return False
        
        print(f"[PASS] GET schedule 1: date={result['date']}, status={result['status']}")
        print(f"  Demand rows: {len(result['demand'])}")
        print(f"  Supply rows: {len(result['supply'])}")
        
        ecg_items = [d for d in result['demand'] if d['entity_name'] == 'ECG']
        ecg_sorted = sorted(ecg_items, key=lambda x: x['hour'])
        h10 = next((d for d in ecg_sorted if d['hour'] == 10), None)
        print(f"  ECG Hour 10: {h10['demand_mw']} MW")
        
        checks = []
        checks.append(("Schedule returned", result is not None))
        checks.append(("Demand list present", len(result['demand']) == 48))
        checks.append(("Supply list present", len(result['supply']) == 42))
        
        nits_items = [d for d in result['demand'] if d['entity_name'] == 'NITS_Total']
        checks.append(("NITS_Total present", len(nits_items) > 0))
        
        checks.append(("ECG Hour 10 value correct", h10 is not None and abs(h10['demand_mw'] - 1808.11) < 0.01))
        checks.append(("Status is draft", result['status'] == 'draft'))
        
        print(f"\nGATE TEST RESULTS (GET endpoint):")
        all_pass = True
        for label, ok in checks:
            print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
            if not ok:
                all_pass = False
        
        print(f"\n  OVERALL: [{'PASS' if all_pass else 'FAIL'}]")
        return all_pass


if __name__ == '__main__':
    success = asyncio.run(test_get())
    sys.exit(0 if success else 1)
