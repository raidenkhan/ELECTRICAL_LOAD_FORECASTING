import sqlite3
db = sqlite3.connect("Backend/loadforecast.db")
cur = db.execute("select name from sqlite_master where type='table' order by name")
for row in cur:
    print(row[0])
db.close()
