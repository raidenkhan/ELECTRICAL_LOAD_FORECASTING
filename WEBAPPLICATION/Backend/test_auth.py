"""
Test script for authentication endpoints
"""
import requests
import json

BASE_URL = "http://127.0.0.1:8000/api/v1"

def test_user_signup():
    """Test user registration"""
    print("\n=== Testing User Signup ===")
    
    signup_data = {
        "email": "testuser@example.com",
        "password": "SecurePassword123!",
        "full_name": "Test User"
    }
    
    response = requests.post(f"{BASE_URL}/users/signup", json=signup_data)
    
    print(f"Status Code: {response.status_code}")
    
    try:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except:
        print(f"Response Text: {response.text}")
    
    return response.status_code == 201

def test_login():
    """Test user login and token generation"""
    print("\n=== Testing Login ===")
    
    login_data = {
        "username": "testuser@example.com",  # OAuth2 uses 'username' field
        "password": "SecurePassword123!"
    }
    
    response = requests.post(
        f"{BASE_URL}/access-token",
        data=login_data,  # Form data, not JSON
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    if response.status_code == 200:
        return response.json()["access_token"]
    return None

def test_get_current_user(token):
    """Test accessing protected route with token"""
    print("\n=== Testing Protected Route (Get Current User) ===")
    
    headers = {
        "Authorization": f"Bearer {token}"
    }
    
    response = requests.get(f"{BASE_URL}/users/me", headers=headers)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200

def test_invalid_token():
    """Test accessing protected route with invalid token"""
    print("\n=== Testing Invalid Token ===")
    
    headers = {
        "Authorization": "Bearer invalid_token_here"
    }
    
    response = requests.get(f"{BASE_URL}/users/me", headers=headers)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 403

if __name__ == "__main__":
    print("=" * 60)
    print("Authentication Endpoints Test Suite")
    print("=" * 60)
    
    # Test 1: User Signup
    signup_success = test_user_signup()
    
    if not signup_success:
        print("\n⚠️  Signup failed or user already exists. Continuing with login test...")
    
    # Test 2: Login
    token = test_login()
    
    if not token:
        print("\n❌ Login failed! Cannot continue with protected route tests.")
        exit(1)
    
    print(f"\n✅ Successfully obtained access token: {token[:20]}...")
    
    # Test 3: Access protected route with valid token
    if test_get_current_user(token):
        print("\n✅ Successfully accessed protected route with valid token")
    else:
        print("\n❌ Failed to access protected route with valid token")
    
    # Test 4: Access protected route with invalid token
    if test_invalid_token():
        print("\n✅ Correctly rejected invalid token")
    else:
        print("\n❌ Should have rejected invalid token")
    
    print("\n" + "=" * 60)
    print("Test Suite Complete!")
    print("=" * 60)
