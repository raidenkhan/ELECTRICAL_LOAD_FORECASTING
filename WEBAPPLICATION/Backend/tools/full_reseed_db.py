import asyncio
import sys
import os
import pandas as pd
import numpy as np
from sqlalchemy.future import select
from sqlalchemy import text
from datetime import datetime

# Add the parent directory to sys.path to enable imports from 'app'
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from app.db.session import AsyncSessionLocal 
from app.db.models.data import ValidatedData, RawDataUpload

async def full_reseed():
    print("=" * 60)
    print("CORE SYSTEM RE-SEED: NAYAGINA-82 FULL HISTORY")
    print("=" * 60)
    
    scada_file = os.path.abspath(os.path.join(parent_dir, "../../EXTRAS/resampled_data_15min.csv"))
    meteo_file = os.path.abspath(os.path.join(parent_dir, "../../openmeteoweather.csv"))
    
    if not os.path.exists(scada_file):
        print(f"Error: {scada_file} not found")
        return

    # 1. Load and Clean Data (Standard GRIDCo logic)
    print("Reading SCADA data...")
    df = pd.read_csv(scada_file)
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])
    df = df.sort_values('DATETIME').reset_index(drop=True)

    # Calculate Total Community Load
    t1 = df['82T1_BANK (MW)'].clip(lower=0)
    t3 = df['82T3_BANK (MW)']
    t4 = df['82T4_BANK (MW)']
    df['load_mw'] = (t1 + t3 + t4).clip(lower=0)
    
    # 2. Merge Weather for high-fidelity priors
    print("Merging weather gradients...")
    try:
        meteo = pd.read_csv(meteo_file, skiprows=3)
        meteo['DATETIME'] = pd.to_datetime(meteo['time'])
        meteo = (meteo.set_index('DATETIME')[['temperature_2m (°C)']]
                     .resample('15min').interpolate(method='linear').reset_index())
        meteo.rename(columns={'temperature_2m (°C)': 'Temp'}, inplace=True)
        df = pd.merge(df, meteo, on='DATETIME', how='left')
        df['Temp'] = df['Temp'].ffill().bfill().fillna(28.0)
    except:
        print("Warning: Weather merge failed, using persistence.")
        df['Temp'] = 28.0

    async with AsyncSessionLocal() as session:
        # 3. Wipe existing data (SQLite compatible)
        print("Wiping existing records...")
        await session.execute(text("DELETE FROM validated_data"))
        await session.execute(text("DELETE FROM raw_data_uploads"))
        await session.execute(text("DELETE FROM validation_reports"))
        # Reset auto-increment for SQLite
        try:
            await session.execute(text("DELETE FROM sqlite_sequence WHERE name IN ('validated_data', 'raw_data_uploads', 'validation_reports')"))
        except:
            pass # sequence table might not exist
        await session.commit()

        # 4. Create master upload record
        upload = RawDataUpload(
            filename="Full_History_2025_06_22.csv",
            file_size_bytes=os.path.getsize(scada_file),
            row_count=len(df),
            status="validated",
            upload_timestamp=datetime.utcnow()
        )
        session.add(upload)
        await session.commit()
        await session.refresh(upload)
        
        # 5. Bulk Insert in chunks
        print(f"Preparing {len(df)} records for injection...")
        records = []
        for _, row in df.iterrows():
            records.append(ValidatedData(
                upload_id=upload.id,
                timestamp=row['DATETIME'],
                total_load_mw=float(row['load_mw']),
                temperature_c=float(row['Temp']),
                voltage_kv=float(row.get('AD2NY_LINE (KV)', 33.0)),
                frequency_hz=float(row.get('FREQ (HZ)', 50.0)),
                is_anomaly=False,
                validation_flags={}
            ))
            
            if len(records) >= 5000:
                session.add_all(records)
                await session.commit()
                print(f"  -> Injected {len(records)} records...")
                records = []
        
        if records:
            session.add_all(records)
            await session.commit()
            print(f"  -> Injected final {len(records)} records.")

    print("\n" + "=" * 60)
    print("SUCCESS: System re-seeded with 1.5 years of history.")
    print("=" * 60)

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(full_reseed())
