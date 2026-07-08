import asyncio
import os
import sys

# Add Backend to path
sys.path.append(os.path.join(os.getcwd(), 'Backend'))

from app.db.session import AsyncSessionLocal
from app.db.models.ecg_history import EcgHistoricalDemand
from sqlalchemy import select, func

async def check():
    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(func.count(EcgHistoricalDemand.id)))
            count = result.scalar()
            print(f"ECG Historical Demand Count: {count}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(check())
