import sqlite3
import os

db_path = r"c:\Users\User\Desktop\project\llm_from_scratch-main\LOADFORECASINGPROJECT\WEBAPPLICATION\Backend\loadforecast.db"

if not os.path.exists(db_path):
    print(f"DB not found at {db_path}")
else:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Delete points where load > 500 MW (safe limit for community load which is normally < 200 MW)
    cursor.execute("DELETE FROM validated_data WHERE total_load_mw > 500")
    deleted = cursor.rowcount
    conn.commit()
    print(f"Successfully deleted {deleted} outlier records.")
    
    # Check new max
    cursor.execute("SELECT MAX(total_load_mw) FROM validated_data")
    new_max = cursor.fetchone()[0]
    print(f"New Maximum Load in DB: {new_max} MW")
    conn.close()
