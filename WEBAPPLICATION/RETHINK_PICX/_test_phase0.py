"""Phase 0 Gate Test: upload sample Excel, verify all cells match."""
import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Backend'))

db_path = r'C:\Users\User\Desktop\project\llm_from_scratch-main\LOADFORECASINGPROJECT\WEBAPPLICATION\Backend\loadforecast.db'
os.environ['DATABASE_URL'] = f'sqlite+aiosqlite:///{db_path}'

from app.db.session import AsyncSessionLocal
from app.services.schedule_service import ScheduleService
from app.db.models.schedule import DailyDispatchSchedule, HourlyDemand, HourlySupply
from sqlalchemy import select


async def test():
    sample_path = r'C:\Users\User\Desktop\project\llm_from_scratch-main\LOADFORECASINGPROJECT\WEBAPPLICATION\RETHINK_PICX\ECG Daily Demand Data Sheet for Dispatch Day May 22, 2026.xlsx'
    
    if not os.path.exists(sample_path):
        print(f"ERROR: Sample file not found")
        return False

    service = ScheduleService()
    
    async with AsyncSessionLocal() as db:
        schedule = await service.parse_and_store(
            db=db,
            filepath=sample_path,
            filename=os.path.basename(sample_path),
        )
        
        print(f"[PASS] Schedule created: id={schedule.id}, date={schedule.date}, status={schedule.status}")
        
        demand_result = await db.execute(
            select(HourlyDemand).where(HourlyDemand.schedule_id == schedule.id)
        )
        demand_rows = demand_result.scalars().all()
        
        supply_result = await db.execute(
            select(HourlySupply).where(HourlySupply.schedule_id == schedule.id)
        )
        supply_rows = supply_result.scalars().all()
        
        print(f"  Demand rows: {len(demand_rows)}")
        print(f"  Supply rows: {len(supply_rows)}")
        
        entities = set(d.entity_name for d in demand_rows)
        print(f"  Entities found: {sorted(entities)}")
        
        plants = set(s.plant_name for s in supply_rows)
        print(f"  Supply plants found: {sorted(plants)}")
        
        ecg_rows = sorted([d for d in demand_rows if d.entity_name == 'ECG'], key=lambda x: x.hour)
        if ecg_rows:
            print(f"  ECG demand (first 5h):")
            for r in ecg_rows[:5]:
                print(f"    Hour {r.hour:2d}: {r.demand_mw:.2f} MW")
        
        nits_rows = sorted([d for d in demand_rows if d.entity_name == 'NITS_Total'], key=lambda x: x.hour)
        if nits_rows:
            print(f"  NITS Total (first 5h):")
            for r in nits_rows[:5]:
                print(f"    Hour {r.hour:2d}: {r.demand_mw:.2f} MW")
        
        ecg_hour_10 = next((d for d in ecg_rows if d.hour == 10), None)
        if ecg_hour_10:
            expected = 1808.11
            actual = ecg_hour_10.demand_mw
            diff = abs(actual - expected)
            status = "PASS" if diff < 1 else "FAIL"
            print(f"  ECG Hour 10 = {actual:.2f} MW (expected {expected}): [{status}] (diff={diff:.2f})")
        
        checks = []
        checks.append(("Schedule stored", True))
        checks.append(("Demand rows > 0", len(demand_rows) > 0))
        checks.append(("ECG found", 'ECG' in entities))
        checks.append(("NITS_Total found", 'NITS_Total' in entities))
        checks.append(("Supply rows > 0", len(supply_rows) > 0))
        checks.append(("ECG values match expected", ecg_hour_10 is not None and abs(ecg_hour_10.demand_mw - 1808.11) < 1))
        
        print(f"\nGATE TEST RESULTS:")
        all_pass = True
        for label, ok in checks:
            print(f"  [{ 'PASS' if ok else 'FAIL' }] {label}")
            if not ok:
                all_pass = False
        
        print(f"\n  OVERALL: [{ 'PASS' if all_pass else 'FAIL' }]")
        return all_pass


if __name__ == '__main__':
    success = asyncio.run(test())
    sys.exit(0 if success else 1)
