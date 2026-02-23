
import pandas as pd
import requests
import os
import sys

# Configuration
BASE_URL = "http://127.0.0.1:8001/api/v1"
DATA_PATH = r"c:\Users\User\Desktop\project\llm_from_scratch-main\LOADFORECASINGPROJECT\resampled_data_15min.csv"

def verify_stlf():
    print("--- STLF Verification Script ---")
    
    # 1. Prepare Data (Last 14 days)
    print(f"Reading data from {DATA_PATH}...")
    try:
        df = pd.read_csv(DATA_PATH, index_col='DATETIME', parse_dates=True)
    except Exception as e:
        print(f"Error reading data: {e}")
        return

    # Take a slice
    sample_df = df.tail(14 * 96).copy() # 2 weeks
    
    # Calculate Total Load (Physics benchmark)
    # Use abs() to ensure it passes the sign convention (loads should be positive)
    sample_df['TOTAL_LOAD_MW'] = (sample_df['82T1_BANK (MW)'] + sample_df['82T3_BANK (MW)'] + sample_df['82T4_BANK (MW)']).abs()
    
    # Map other columns for physics validation if they exist
    # NY6ZA Flow proxy (line1_mw)
    if 'NY6ZA_LINE (MW)' in sample_df.columns:
        sample_df['line1_mw'] = sample_df['NY6ZA_LINE (MW)'].abs()
    
    # Generation proxy (line2_mw)
    if 'T2_GEN (MW)' in sample_df.columns:
        sample_df['line2_mw'] = sample_df['T2_GEN (MW)'].abs()
        
    # Reset index and rename DATETIME to timestamp
    sample_df = sample_df.reset_index()
    sample_df = sample_df.rename(columns={'DATETIME': 'timestamp'})
    
    # Save to temp csv
    temp_csv = "data/temp_verification_data.csv"
    os.makedirs("data", exist_ok=True)
    sample_df.to_csv(temp_csv)
    print(f"Created temp payload: {temp_csv} ({len(sample_df)} rows)")
    
    # 2. Upload Data
    print("Uploading data to API...")
    url = f"{BASE_URL}/data/upload"
    try:
        with open(temp_csv, 'rb') as f:
            files = {'file': (temp_csv, f, 'text/csv')}
            response = requests.post(url, files=files)
            
        if response.status_code == 200:
            print("Upload Success:", response.json())
        else:
            print("Upload Failed:", response.text)
            return
    except Exception as e:
        print(f"API request failed: {e}")
        return

    # 3. Request Forecast
    print("\nRequesting STLF Forecast (24h)...")
    url = f"{BASE_URL}/forecast/stlf"
    payload = {"horizon_hours": 24}
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            data = response.json()
            print("Forecast Success!")
            print(f"Forecast ID: {data['forecast_id']}")
            print(f"First Prediction: {data['timestamps'][0]} -> {data['forecast_mw'][0]:.2f} MW")
            print(f"Uncertainty P10: {data['p10'][0]:.2f} MW")
            print(f"Uncertainty P90: {data['p90'][0]:.2f} MW")
        else:
            print("Forecast Failed:", response.text)
    except Exception as e:
        print(f"Forecast request failed: {e}")

if __name__ == "__main__":
    verify_stlf()
