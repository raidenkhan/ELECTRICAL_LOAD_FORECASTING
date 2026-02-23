
import asyncio
from app.db.session import AsyncSessionLocal
from app.db.models.user import User
from app.core.security import get_password_hash

async def create_user():
    async with AsyncSessionLocal() as db:
        user = User(
            email='admin@example.com', 
            full_name='Admin User', 
            hashed_password=get_password_hash('admin123'), 
            is_superuser=True
        )
        db.add(user)
        try:
            await db.commit()
            print('User created')
        except Exception as e:
            print(f'Error creating user: {e}')

if __name__ == '__main__':
    asyncio.run(create_user())
