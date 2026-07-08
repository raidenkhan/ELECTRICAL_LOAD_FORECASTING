import sqlite3
db = sqlite3.connect('Backend/loadforecast.db')
tables = db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
for t in tables:
    print(t[0])
db.close()
