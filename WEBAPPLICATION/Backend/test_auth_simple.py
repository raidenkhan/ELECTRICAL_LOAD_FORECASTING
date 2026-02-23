"""
Simple authentication test using httpx (already installed)
"""
import asyncio
import httpx
import json

BASE_URL = "http://127.0.0.1:8000/api/v1"

async def test_signup():
    async with httpx.AsyncClient() as client:
        print("\n=== Testing User Signup ===")
        
        signup_data = {
            "email": "testuser@example.com",
            "password": "SecurePassword123!",
            "full_name": "Test User"
        }
        
        try:
            response = await client.post(f"{BASE_URL}/users/signup", json=signup_data)
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text}")
            return response.status_code == 201
        except Exception as e:
            print(f"Error: {e}")
            return False

async def test_login():
    async with httpx.AsyncClient() as client:
        print("\n=== Testing Login ===")
        
        login_data = {
            "username": "testuser@example.com",
            "password": "SecurePassword123!"
        }
        
        try:
            response = await client.post(
                f"{BASE_URL}/access-token",
                data=login_data
            )
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text}")
            
            if response.status_code == 200:
                return response.json()["access_token"]
            return None
        except Exception as e:
            print(f"Error: {e}")
            return None

async def test_get_user(token):
    async with httpx.AsyncClient() as client:
        print("\n=== Testing Get Current User ===")
        
        headers = {"Authorization": f"Bearer {token}"}
        
        try:
            response = await client.get(f"{BASE_URL}/users/me", headers=headers)
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text}")
            return response.status_code == 200
        except Exception as e:
            print(f"Error: {e}")
            return False

async def main():
    print("=" * 60)
    print("Authentication Endpoints Test")
    print("=" * 60)
    
    # Test signup
    signup_ok = await test_signup()
    if not signup_ok:
        print("\nSignup failed or user exists. Continuing...")
    
    # Test login
    token = await test_login()
    if not token:
        print("\nLogin failed!")
        return
    
    print(f"\nToken obtained: {token[:30]}...")
    
    # Test protected route
    if await test_get_user(token):
        print("\n✓ Successfully accessed protected route!")
    else:
        print("\n✗ Failed to access protected route")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
