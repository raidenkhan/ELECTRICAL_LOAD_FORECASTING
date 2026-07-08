"""Entry point for PyInstaller-bundled GRIDCo Load Forecaster.

1. Creates writable data/ directory next to the .exe
2. Runs Alembic migrations to create/update the SQLite database
3. Seeds ecg_historical_demand from full 2018-2026 CSV if table is empty
4. Opens browser to http://localhost:8000
5. Starts uvicorn server
"""
import os, sys, webbrowser
from pathlib import Path

if getattr(sys, 'frozen', False):
    exe_dir = Path(sys.executable).parent
    bundle_dir = Path(sys._MEIPASS)
else:
    exe_dir = Path(__file__).resolve().parent
    bundle_dir = exe_dir

data_dir = exe_dir / "data"
data_dir.mkdir(parents=True, exist_ok=True)

os.chdir(str(exe_dir))

print("Running database migrations...")
try:
    from alembic.config import Config
    from alembic import command
    alembic_cfg = Config(str(bundle_dir / "alembic.ini"))
    alembic_cfg.set_main_option("script_location", str(bundle_dir / "migrations"))
    command.upgrade(alembic_cfg, "head")
    print("Migrations complete.")
except Exception as e:
    print(f"WARNING: Migration failed: {e}")
    print("The application may still work if the database already exists.")

db_path = data_dir / "loadforecast.db"
csv_path = bundle_dir / "tools" / "dl_forecast" / "ecg_demand_2018_2026.csv"
if csv_path.exists():
    from sqlalchemy import create_engine, text
    seed_engine = create_engine(f"sqlite:///{db_path}", echo=False)
    with seed_engine.connect() as conn:
        row = conn.execute(text("SELECT COUNT(*) FROM ecg_historical_demand")).scalar()
    if row == 0:
        print("Seeding historical demand data from CSV (70,228 rows)...")
        import pandas as pd
        from datetime import datetime
        df = pd.read_csv(csv_path)
        df["created_at"] = datetime.utcnow()
        df.to_sql("ecg_historical_demand", seed_engine, if_exists="append", index=False)
        print(f"Seeded {len(df)} rows of historical data.")
    else:
        print(f"Historical data already present ({row} rows).")
    seed_engine.dispose()
else:
    print(f"WARNING: Seed CSV not found at {csv_path}")

webbrowser.open("http://localhost:8000")

print("Starting GRIDCo Load Forecaster on http://localhost:8000")
import uvicorn
from app.main import app
uvicorn.run(app, host="0.0.0.0", port=8000)
