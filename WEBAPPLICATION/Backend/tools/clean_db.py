
import asyncio
import sys
import os
from sqlalchemy import text

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from app.db.session import AsyncSessionLocal

async def main():
    print("Cleaning database...")
    async with AsyncSessionLocal() as session:
        # Delete all records from ValidatedData
        # We might need to delete RawDataUpload too if there are FK constraints, 
        # but ValidatedData is the child usually? 
        # Actually ValidatedData -> upload_id -> RawDataUpload. 
        # So deleting ValidatedData is fine.
        
        # Using execute with text for direct delete if models aren't perfect, 
        # but ORM way is better if we want to be safe. 
        # Let's use text for speed/simplicity in this tool.
        
        await session.execute(text("DELETE FROM validated_data"))
        await session.execute(text("DELETE FROM raw_data_uploads")) # Clean parent too to be fresh
        
        await session.commit()
        print("Database cleaned.")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
