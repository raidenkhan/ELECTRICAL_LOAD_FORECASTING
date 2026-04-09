import pandas as pd
import numpy as np
import os
import sys
from sqlalchemy import create_engine, text
import datetime

# Add project root to path to access app config if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def restore_factory_baseline():
    """
    Safely wipes the database and restores the original Community Load dataset.
    """
    # DATABASE CONFIG
    # Try to load from environment or fall back to local sqlite
    import dotenv
    dotenv.load_dotenv()
    
    DB_URL = os.getenv("DATABASE_URL", "sqlite:///./loadforecast.db")
    # Convert async driver to sync for this script if necessary
    if "postgresql+asyncpg" in DB_URL:
        DB_URL = DB_URL.replace("postgresql+asyncpg", "postgresql")
    elif "sqlite+aiosqlite" in DB_URL:
        DB_URL = DB_URL.replace("sqlite+aiosqlite", "sqlite")
    
    ORIGINAL_CSV = r"c:\Users\User\Desktop\project\llm_from_scratch-main\LOADFORECASINGPROJECT\resampled_data_15min.csv"
    
    if not os.path.exists(ORIGINAL_CSV):
        print(f"Error: Original dataset not found at {ORIGINAL_CSV}")
        return

    print(f"--- SYSTEM RESET INITIATED (DB: {DB_URL}) ---")
    
    try:
        # 1. Connect to DB
        engine = create_engine(DB_URL)
        is_sqlite = "sqlite" in DB_URL
        
        # 2. Extract and Map Original Data
        print("Reading community load baseline...")
        df_raw = pd.read_csv(ORIGINAL_CSV, nrows=2000)
        
        # Mapping Logic
        df = pd.DataFrame()
        df['timestamp'] = pd.to_datetime(df_raw['DATETIME'])
        bank_cols = [col for col in df_raw.columns if '_BANK (MW)' in col]
        df['total_load_mw'] = df_raw[bank_cols].sum(axis=1)
        df['voltage_kv'] = df_raw.get('AD2NY_LINE (KV)', 33.0)
        df['frequency_hz'] = df_raw.get('FREQ (HZ)', 50.0)
        df['temperature_c'] = df_raw.get('82T1_TEMPERATURE_WDG_1 (VALUE)', 25.0)
        df['current_a'] = df_raw.get('82T1_BANK (A)', 150.0)
        df['upload_id'] = 0
        df['is_anomaly'] = False
        df['validation_flags'] = '{}'
        
        # 3. Wipe Existing Data
        print("Cleaning system tables...")
        with engine.begin() as conn:
            if is_sqlite:
                conn.execute(text("DELETE FROM validated_data;"))
                conn.execute(text("DELETE FROM raw_data_uploads;"))
                conn.execute(text("DELETE FROM validation_reports;"))
                # Reset auto-increment (handle if sqlite_sequence doesn't exist)
                try:
                    conn.execute(text("DELETE FROM sqlite_sequence WHERE name IN ('validated_data', 'raw_data_uploads', 'validation_reports');"))
                except Exception:
                    pass 
            else:
                conn.execute(text("TRUNCATE TABLE validated_data RESTART IDENTITY CASCADE;"))
                conn.execute(text("TRUNCATE TABLE raw_data_uploads RESTART IDENTITY CASCADE;"))
                conn.execute(text("TRUNCATE TABLE validation_reports RESTART IDENTITY CASCADE;"))
        
        # 4. Bulk Insert Baseline
        print(f"Restoring {len(df)} community load points...")
        df.to_sql('validated_data', con=engine, if_exists='append', index=False)
        
        # Insert a marker for the restore
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO raw_data_uploads (filename, file_size_bytes, row_count, status, upload_timestamp)
                VALUES ('System Baseline (Community Load)', 0, :row_count, 'validated', CURRENT_TIMESTAMP)
            """), {"row_count": len(df)})

        print("--- REVERT COMPLETE: Community Load Baseline Active ---")
        
    except Exception as e:
        print(f"FAILED TO RESTORE SYSTEM: {str(e)}")

if __name__ == "__main__":
    restore_factory_baseline()
