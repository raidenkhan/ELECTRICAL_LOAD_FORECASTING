import asyncio
import sys
import os

# Add backend to path
sys.path.append(os.getcwd())

from app.core.security import verify_password, get_password_hash
from app.db.session import AsyncSessionLocal
from app.services.user_service import get_user_by_email
from sqlalchemy import text

async def main():
    print("Debugging Password Verification")
    
    # 1. Test local hashing
    pwd = "testpassword123"
    hashed = get_password_hash(pwd)
    print(f"Generated hash for '{pwd}': {hashed}")
    valid = verify_password(pwd, hashed)
    print(f"Local verification result: {valid}")
    
    if not valid:
        print("CRITICAL: Local hash verification failed immediately!")
        return

    # 2. Check DB user
    email = "testuser@example.com"
    print(f"\nChecking user '{email}' in database...")
    
    # Direct DB access to get hash
    async with AsyncSessionLocal() as session:
        result = await session.execute(text(f"SELECT hashed_password FROM users WHERE email = '{email}'"))
        row = result.fetchone()
        
        if not row:
            print("User not found in DB!")
            return
            
        db_hash = row[0]
        print(f"Hash in DB: {db_hash}")
        
        # Verify
        db_valid = verify_password(pwd, db_hash)
        print(f"Verification against DB hash: {db_valid}")

if __name__ == "__main__":
    asyncio.run(main())
