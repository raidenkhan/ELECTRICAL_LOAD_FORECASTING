"""
Export Seed Data — GridForecast Pro
=====================================
Run this ONCE from the Backend directory on your dev machine to produce
data/ecg_seed.csv, which will be shipped inside the release ZIP and
auto-loaded by startup.sh on a fresh GRIDCo installation.

Usage:
    cd Backend
    python tools/export_seed.py

Output:
    data/ecg_seed.csv  (last 250 hours of ecg_historical_demand)
"""

import asyncio
import csv
import os
import sys

# Allow running from Backend/ directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import select, text
from app.db.session import AsyncSessionLocal
from app.db.models.ecg_history import EcgHistoricalDemand

HOURS_TO_EXPORT = 250   # 168 minimum needed + 82 buffer
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "ecg_seed.csv")


async def export():
    print(f"Connecting to database...")
    async with AsyncSessionLocal() as db:
        # Count rows first
        count_result = await db.execute(text("SELECT COUNT(*) FROM ecg_historical_demand"))
        total = count_result.scalar()
        print(f"Total rows in ecg_historical_demand: {total}")

        if total == 0:
            print("ERROR: Database is empty — nothing to export. Run the app and upload data first.")
            sys.exit(1)

        # Fetch the last HOURS_TO_EXPORT rows ordered by date+hour ascending
        stmt = (
            select(EcgHistoricalDemand)
            .order_by(EcgHistoricalDemand.date.desc(), EcgHistoricalDemand.hour.desc())
            .limit(HOURS_TO_EXPORT)
        )
        result = await db.execute(stmt)
        rows = list(reversed(result.scalars().all()))

    print(f"Exporting {len(rows)} rows...")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "hour", "demand_mw", "temperature_c", "is_holiday"])
        for row in rows:
            writer.writerow([
                row.date.isoformat(),
                row.hour,
                round(float(row.demand_mw), 4),
                round(float(row.temperature_c), 2) if row.temperature_c is not None else 28.0,
                int(row.is_holiday or 0),
            ])

    print(f"\n[OK] Seed data exported to: {OUTPUT_PATH}")
    print(f"     Date range: {rows[0].date} H{rows[0].hour} -> {rows[-1].date} H{rows[-1].hour}")
    print(f"     Row count : {len(rows)}")
    print(f"\nNow re-run make_zip.ps1 to bundle this file into gridforecast_release.zip")


if __name__ == "__main__":
    asyncio.run(export())
