import sqlite3, os

dbs = [
    "loadforecast.db",
    "Backend/loadforecast.db",
    "Backend/app.db",
]
for f in dbs:
    path = os.path.join(os.path.dirname(__file__) or ".", f)
    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)) or ".", f)
    print(f"\n=== {f} ({os.path.getsize(path)} bytes) ===" if os.path.exists(path) else f"\n=== {f} NOT FOUND ===")
    if os.path.exists(path):
        db = sqlite3.connect(path)
        tables = db.execute("select name from sqlite_master where type='table' order by name").fetchall()
        for t in tables:
            count = db.execute(f"select count(*) from \"{t[0]}\"").fetchone()[0]
            print(f"  {t[0]}: {count} rows")
        db.close()
