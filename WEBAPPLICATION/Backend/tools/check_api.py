
import requests
import sys

URL = "http://127.0.0.1:8001/api/v1/openapi.json"

try:
    print(f"Checking {URL}...")
    resp = requests.get(URL, timeout=5)
    print(f"Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        stlf_path = "/api/v1/forecast/stlf"
        if stlf_path in data.get("paths", {}):
            print(f"SUCCESS: {stlf_path} found!")
        else:
            print(f"FAILURE: {stlf_path} NOT found.")
            print("Accessible paths:", list(data.get("paths", {}).keys()))
    else:
        print(f"Content: {resp.text[:200]}")
except Exception as e:
    print(f"Error: {e}")
