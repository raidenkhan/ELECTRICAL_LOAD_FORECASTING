
import pandas as pd
import requests
import os

# Configuration
BASE_URL = "http://127.0.0.1:8001/api/v1"

def verify_ltlf():
    print("--- LTLF Verification Script ---")
    
    # LTLF doesn't necessarily need an upload if data is already in DB, 
    # but we'll assume we want to trigger it.
    # LTLF usually predicts days ahead.
    
    print("\nRequesting LTLF Forecast (7 days)...")
    url = f"{BASE_URL}/forecast/ltlf"
    # LTLF might expect horizon_hours or days
    payload = {"horizon_hours": 168} # 7 days
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            data = response.json()
            print("Forecast Success!")
            print(f"Forecast ID: {data['forecast_id']}")
            print(f"First Prediction: {data['timestamps'][0]} -> {data['forecast_mw'][0]:.2f} MW")
            print(f"Number of steps: {len(data['forecast_mw'])}")
        else:
            print("Forecast Failed:", response.text)
    except Exception as e:
        print(f"Forecast request failed: {e}")

if __name__ == "__main__":
    verify_ltlf()
