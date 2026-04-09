import sqlite3
import pandas as pd
import os

db_path = r"c:\Users\User\Desktop\project\llm_from_scratch-main\LOADFORECASINGPROJECT\WEBAPPLICATION\Backend\loadforecast.db"

if not os.path.exists(db_path):
    print(f"DB not found at {db_path}")
else:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT id, timestamp, total_load_mw FROM validated_data ORDER BY total_load_mw DESC LIMIT 20", conn)
    print("top 20 load values:")
    print(df)
    
    stats = pd.read_sql_query("SELECT COUNT(*), AVG(total_load_mw), MAX(total_load_mw), MIN(total_load_mw) FROM validated_data", conn)
    print("\nOverall Stats:")
    print(stats)
    conn.close()
