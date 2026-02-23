import requests

print("Verifying login for testuser@example.com...")
try:
    response = requests.post(
        "http://localhost:8000/api/v1/access-token",
        data={"username": "testuser@example.com", "password": "testpassword123"}
    )
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 200:
        print("SUCCESS: Backend login works.")
    else:
        print("FAILURE: Backend login failed.")
except Exception as e:
    print(f"Error: {e}")
