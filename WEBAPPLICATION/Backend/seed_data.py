
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add current directory to path so we can import app modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.db.session import AsyncSessionLocal
from app.db.models.data import ValidatedData, RawDataUpload

async def seed_data():
    print("Starting data seeding...")
    
    csv_path = "../../../LOADFORECASINGPROJECT/resampled_data_15min.csv"
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    # Read CSV (only needed columns to save memory)
    print("Reading CSV...")
    df = pd.read_csv(csv_path)
    
    # Parse datetime
    if 'DATETIME' in df.columns:
        df['timestamp'] = pd.to_datetime(df['DATETIME'])
    else:
        print("Error: DATETIME column not found")
        return
        
    df.sort_values('timestamp', inplace=True)
    
    # Cutoff for valid data
    cutoff_date = pd.Timestamp("2025-05-01")
    df = df[df['timestamp'] < cutoff_date]
    
    # Take last 10,000 rows (approx 3.5 months) of VALID data
    df = df.tail(10000).copy()
    print(f"Processing last {len(df)} rows (valid data before {cutoff_date.date()})...")

    # Calculate Total Load
    # Community Load = T1 + T3 + T4 (exclude T2 which is generation)
    # T1: "82T1_BANK (MW)"
    # T3: "82T3_BANK (MW)"
    # T4: "82T4_BANK (MW)"
    
    t1_col = "82T1_BANK (MW)"
    t2_col = "82T2_BANK (MW)"
    t3_col = "82T3_BANK (MW)"
    t4_col = "82T4_BANK (MW)"
    
    # Fill NaNs with 0 for calculation
    for col in [t1_col, t3_col, t4_col]:
        if col in df.columns:
            df[col] = df[col].fillna(0)
            
    df['total_load_mw'] = df[t1_col] + df[t3_col] + df[t4_col]
    
    # Map other columns
    # LINE1 -> NY6ZA_LINE (MW)
    # LINE2 -> T2 Generation (82T2_BANK (MW))
    # LINE3 -> Reserved (maybe T1)
    
    line1_col = "NY6ZA_LINE (MW)"
    freq_col = "FREQ (HZ)"
    
    # Temperature: find column starting with "82T1_TEMPERATURE_WDG_1"
    temp_col = None
    for c in df.columns:
        if c.startswith("82T1_TEMPERATURE_WDG_1"):
            temp_col = c
            break
            
    # Voltage/Current (Proxies)
    volt_col = "NY6ZA_LINE (KV)"
    curr_col = "NY6ZA_LINE (A)"

    async with AsyncSessionLocal() as session:
        # 1. Create dummy upload record
        print("Creating upload record...")
        upload = RawDataUpload(
            filename="resampled_data_15min_SEED.csv",
            file_size_bytes=0,
            row_count=len(df),
            status="validated",
            upload_timestamp=datetime.utcnow()
        )
        session.add(upload)
        await session.commit()
        await session.refresh(upload)
        
        upload_id = upload.id
        print(f"Created Upload ID: {upload_id}")
        
        # 2. Bulk insert validated data
        print("Preparing records...")
        records = []
        for idx, row in df.iterrows():
            record = ValidatedData(
                upload_id=upload_id,
                timestamp=row['timestamp'],
                total_load_mw=float(row['total_load_mw']),
                line1_mw=float(row.get(line1_col, 0.0)),
                line2_mw=float(row.get(t2_col, 0.0)), # T2 is generation
                line3_mw=float(row.get(t1_col, 0.0)), # Using T1 as Line 3 for now
                voltage_kv=float(row.get(volt_col, 0.0)),
                current_a=float(row.get(curr_col, 0.0)),
                temperature_c=float(row.get(temp_col, 0.0)) if temp_col else 0.0,
                frequency_hz=float(row.get(freq_col, 50.0)),
                is_anomaly=False,
                validation_flags={}
            )
            records.append(record)
            
        print(f"Inserting {len(records)} records...")
        # Chunk insertion to avoid parameter limits
        chunk_size = 1000
        for i in range(0, len(records), chunk_size):
            chunk = records[i:i+chunk_size]
            session.add_all(chunk)
            await session.commit()
            print(f"Inserted chunk {i//chunk_size + 1}/{(len(records)-1)//chunk_size + 1}")
            
    print("Seeding complete!")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(seed_data())
