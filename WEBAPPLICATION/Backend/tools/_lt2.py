import sqlite3, os
f = "Backend/loadforecast.db"
print("exists:", os.path.exists(f))
db = sqlite3.connect(f)
cur = db.execute("select name from sqlite_master where type='table' order by name")
for row in cur:
    print("TABLE:", row[0])
db.close()
