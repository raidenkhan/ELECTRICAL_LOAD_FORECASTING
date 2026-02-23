import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.future import select
from sqlalchemy.pool import NullPool
from app.core.config import settings
from app.db.models.user import User

async def verification():
    print(f"DEBUG: DATABASE_URL is: {settings.DATABASE_URL}")
    
    engine = create_async_engine(
        settings.DATABASE_URL,
        echo=True,
        poolclass=NullPool,
        future=True
    )
    
    print("DEBUG: Engine created")
    
    try:
        async with engine.connect() as conn:
            print("DEBUG: Connection successful")
            result = await conn.execute(select(1))
            print(f"DEBUG: Select 1 result: {result.scalar()}")
            
        async with AsyncSession(engine) as session:
            print("DEBUG: Session created")
            result = await session.execute(select(User))
            users = result.scalars().all()
            print(f"DEBUG: Found {len(users)} users")
            
    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        await engine.dispose()

if __name__ == "__main__":
    asyncio.run(verification())
