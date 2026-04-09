import httpx
import sys
import os

def test_api_upload():
    url = "http://localhost:8000/api/v1/data/upload"
    file_path = "test_upload.csv"
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found")
        return

    print(f"Uploading {file_path} to {url}...")
    
    with open(file_path, "rb") as f:
        files = {"file": (file_path, f, "text/csv")}
        try:
            response = httpx.post(url, files=files, timeout=30.0)
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.json()}")
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "validated":
                    print("\nAPI TEST PASSED: File uploaded and validated successfully!")
                else:
                    print(f"\nAPI TEST FAILED: Status is {data.get('status')}")
                    print(f"Message: {data.get('message')}")
            else:
                print(f"\nAPI TEST FAILED: Server returned {response.status_code}")
        except Exception as e:
            print(f"\nAPI TEST ERROR: {str(e)}")

if __name__ == "__main__":
    test_api_upload()
