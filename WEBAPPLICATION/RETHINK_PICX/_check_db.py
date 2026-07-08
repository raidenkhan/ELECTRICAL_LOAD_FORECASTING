import sqlite3
conn = sqlite3.connect(r'C:\Users\User\Desktop\project\llm_from_scratch-main\LOADFORECASINGPROJECT\WEBAPPLICATION\Backend\loadforecast.db')
c = conn.execute('SELECT version_num FROM alembic_version')
print('Alembic version:', c.fetchone())
c = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
print('Tables:', [t[0] for t in c.fetchall()])
conn.close()
