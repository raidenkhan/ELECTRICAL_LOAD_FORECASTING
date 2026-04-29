import asyncio
import sys
import os
import pandas as pd
from sqlalchemy.future import select
from sqlalchemy import func

# Add the parent directory to sys.path to enable imports from 'app'
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from app.db.session import AsyncSessionLocal 
from app.db.models.data import ValidatedData

async def check_db_range():
    async with AsyncSessionLocal() as session:
        # 1. Get Count and Date Range
        count_stmt = select(func.count(ValidatedData.id))
        min_date_stmt = select(func.min(ValidatedData.timestamp))
        max_date_stmt = select(func.max(ValidatedData.timestamp))
        
        count = (await session.execute(count_stmt)).scalar()
        min_date = (await session.execute(min_date_stmt)).scalar()
        max_date = (await session.execute(max_date_stmt)).scalar()
        
        # 2. Calculate Cleaned Mean (excluding outages < 25MW)
        mean_stmt = select(func.avg(ValidatedData.total_load_mw)).where(ValidatedData.total_load_mw >= 25.0)
        clean_mean = (await session.execute(mean_stmt)).scalar()
        
        # 3. Check for specific high peaks
        peaks_stmt = select(func.count(ValidatedData.id)).where(ValidatedData.total_load_mw > 120.0)
        peaks_count = (await session.execute(peaks_stmt)).scalar()

        print("-" * 60)
        print("DATABASE STATE (ValidatedData Table)")
        print("-" * 60)
        print(f"Total Records:      {count}")
        print(f"Start Date:         {min_date}")
        print(f"End Date:           {max_date}")
        print(f"Cleaned Mean (>25MW): {float(clean_mean or 0):.2f} MW")
        print(f"Records > 120 MW:   {peaks_count}")
        print("-" * 60)

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(check_db_range())
