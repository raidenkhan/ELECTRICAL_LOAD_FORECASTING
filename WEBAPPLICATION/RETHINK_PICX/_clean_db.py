import sqlite3
db = r'C:\Users\User\Desktop\project\llm_from_scratch-main\LOADFORECASINGPROJECT\WEBAPPLICATION\Backend\loadforecast.db'
conn = sqlite3.connect(db)
conn.execute("DELETE FROM hourly_demand")
conn.execute("DELETE FROM hourly_supply")
conn.execute("DELETE FROM daily_dispatch_schedules")
conn.commit()
conn.close()
print("Cleaned dispatch schedule tables")
