
import asyncio
import sys
import os

# Add the parent directory to sys.path to enable imports from 'app'
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Correct import based on Step 73 analysis
from app.db.session import AsyncSessionLocal 
from app.db.models.data import ValidatedData
from sqlalchemy.future import select
from sqlalchemy import desc

async def main():
    async with AsyncSessionLocal() as session:
        # Fetch the last 20 records
        stmt = select(ValidatedData).order_by(desc(ValidatedData.timestamp)).limit(20)
        result = await session.execute(stmt)
        data = result.scalars().all()
        
        print(f"Found {len(data)} records in DB.")
        print("-" * 120)
        print(f"{'Timestamp':<25} | {'Total (MW)':<15} | {'L1 (MW)':<12} | {'L2 (MW)':<12} | {'L3 (MW)':<12}")
        print("-" * 120)
        
        for row in data:
            print(f"{str(row.timestamp):<25} | {str(row.total_load_mw):<15} | {str(row.line1_mw):<12} | {str(row.line2_mw):<12} | {str(row.line3_mw):<12}")

if __name__ == "__main__":
    try:
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(main())
    except Exception as e:
        print(f"Error: {e}")
