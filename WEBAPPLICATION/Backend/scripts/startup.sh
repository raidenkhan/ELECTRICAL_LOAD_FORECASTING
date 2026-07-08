#!/bin/bash
set -e

echo "=== GridCo Load Forecasting API — Startup ==="

# 1. Run database migrations
echo ">> Running database migrations..."
alembic upgrade head

# 2. Auto-seed historical ECG data if the table is empty
SEED_FILE="/app/data/ecg_seed.csv"
if [ -f "$SEED_FILE" ]; then
    echo ">> Checking ECG history table..."
    ROW_COUNT=$(python3 - <<'EOF'
import asyncio, sys
from sqlalchemy import text
from app.db.session import AsyncSessionLocal

async def count():
    async with AsyncSessionLocal() as db:
        result = await db.execute(text("SELECT COUNT(*) FROM ecg_historical_demand"))
        print(result.scalar())

asyncio.run(count())
EOF
    )
    echo "   ECG history rows found: $ROW_COUNT"

    if [ "$ROW_COUNT" -lt "168" ]; then
        echo ">> Importing seed data from $SEED_FILE..."
        python3 - <<EOF
import asyncio, csv, sys
from datetime import date
from app.db.session import AsyncSessionLocal
from app.db.models.ecg_history import EcgHistoricalDemand
from sqlalchemy import text

SEED_FILE = "/app/data/ecg_seed.csv"

async def seed():
    async with AsyncSessionLocal() as db:
        with open(SEED_FILE, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = [
                EcgHistoricalDemand(
                    date=date.fromisoformat(r["date"]),
                    hour=int(r["hour"]),
                    demand_mw=float(r["demand_mw"]),
                    temperature_c=float(r["temperature_c"]),
                    is_holiday=bool(int(r.get("is_holiday", 0))),
                )
                for r in reader
            ]
        db.add_all(rows)
        await db.commit()
        print(f"Seeded {len(rows)} ECG history rows from CSV.")

asyncio.run(seed())
EOF
        echo ">> Seed import complete."
    else
        echo ">> ECG history already populated — skipping seed import."
    fi
else
    echo ">> No seed file found at $SEED_FILE — skipping (GRIDCo engineers must upload data via the dashboard)."
fi

# 3. Verify model checkpoints exist
echo ">> Verifying model checkpoints..."
CHECKPOINT_COUNT=$(ls -1 models/dlinear/h10_Fold_*.pt 2>/dev/null | wc -l || echo 0)
if [ "$CHECKPOINT_COUNT" -gt 0 ]; then
    echo "   Found $CHECKPOINT_COUNT DLinear checkpoints — model ready."
else
    echo "   WARNING: No DLinear checkpoints found in models/dlinear/. Run tools/retrain_dlinear.py first."
fi

if [ -f models/weighted_trend_engine.joblib ]; then
    echo "   WT+DOW model found — ready."
else
    echo "   WARNING: WT+DOW model not found at models/weighted_trend_engine.joblib."
fi

# 4. Start the application
echo ">> Starting uvicorn..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
