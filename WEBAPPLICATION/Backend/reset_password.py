import asyncio
import sys
import os

# Add backend to path
sys.path.append(os.getcwd())

from app.core.security import get_password_hash, verify_password
from app.db.session import AsyncSessionLocal
from sqlalchemy import text

async def main():
    email = "testuser@example.com"
    new_password = "testpassword123"
    
    print(f"Resetting password for {email} to '{new_password}'...")
    
    new_hash = get_password_hash(new_password)
    print(f"New Hash: {new_hash}")
    
    async with AsyncSessionLocal() as session:
        # Update password
        await session.execute(
            text(f"UPDATE users SET hashed_password = '{new_hash}' WHERE email = '{email}'")
        )
        await session.commit()
        print("Password updated in DB.")
        
        # Verify
        result = await session.execute(text(f"SELECT hashed_password FROM users WHERE email = '{email}'"))
        row = result.fetchone()
        db_hash = row[0]
        
        valid = verify_password(new_password, db_hash)
        print(f"Verification Check: {valid}")
        
        if valid:
            print("SUCCESS: Password reset and verified!")
        else:
            print("FAILURE: Verification failed even after reset!")

if __name__ == "__main__":
    asyncio.run(main())
