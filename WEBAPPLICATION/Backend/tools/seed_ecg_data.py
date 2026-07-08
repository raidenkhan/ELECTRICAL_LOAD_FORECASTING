"""Seed 12+ months of synthetic ECG hourly demand data for Phase 2 training.

Generates realistic hourly ECG demand based on the actual May 22, 2026
dispatch profile, with day-of-week, seasonal, temperature, and noise variation.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Backend'))

import asyncio
import numpy as np
import pandas as pd
from datetime import date, timedelta
from sqlalchemy import select

from app.db.session import AsyncSessionLocal
from app.db.models.ecg_history import EcgHistoricalDemand
from app.core.logging import get_logger

logger = get_logger(__name__)

# Base profile from real May 22 dispatch: ECG hourly demand (MW)
# Extracted from the uploaded sample Excel
ECG_BASE_PROFILE = np.array([
    1560, 1498, 1440, 1402, 1385, 1400, 1500, 1680,   # 01-08
    1840, 1950, 2010, 2030, 2000, 1970, 1950, 2000,   # 09-16
    2080, 2100, 2050, 1960, 1880, 1820, 1740, 1640,   # 17-24
], dtype=float)

# Seasonal adjustment per month (peak in dry season Mar-May, trough in rainy Jun-Sep)
SEASONAL_FACTOR = {
    1: 0.97, 2: 0.98, 3: 1.02, 4: 1.05, 5: 1.03,
    6: 0.96, 7: 0.93, 8: 0.92, 9: 0.94,
    10: 0.97, 11: 0.99, 12: 0.98,
}

# DOW multiplier: Mon=1, Tue=1, Wed=1, Thu=1, Fri=1.02, Sat=0.92, Sun=0.88
DOW_FACTOR = {0: 1.00, 1: 1.00, 2: 1.00, 3: 1.00, 4: 1.02, 5: 0.92, 6: 0.88}

# Temperature sensitivity: ~1.5% per °C above/below 28°C mean
TEMP_SENSITIVITY = 0.015

# Ghana holidays (fixed + approximate)
HOLIDAYS = {
    "2025-01-01", "2025-03-06", "2025-04-18", "2025-04-21", "2025-05-01",
    "2025-07-01", "2025-08-04", "2025-09-21", "2025-12-01", "2025-12-25", "2025-12-26",
    "2026-01-01", "2026-03-06", "2026-04-03", "2026-04-06", "2026-05-01",
    "2026-05-25", "2026-07-01", "2026-08-03", "2026-09-21", "2026-12-01",
    "2026-12-25", "2026-12-26",
}


def generate_ghana_temp(date_obj: date, hour: int) -> float:
    """Generate realistic Ghana temperature for a given date and hour."""
    month = date_obj.month
    # Base temp varies by season
    if month in [3, 4, 5]:
        base = 32.0  # Hot dry season
    elif month in [6, 7, 8, 9]:
        base = 27.0  # Rainy season
    elif month in [10, 11]:
        base = 29.0  # Transition
    else:
        base = 28.0  # Harmattan
    # Diurnal cycle: coolest at 06:00, hottest at 14:00
    diurnal = -4.0 * np.cos(2 * np.pi * (hour - 6) / 24)
    noise = np.random.normal(0, 1.0)
    return base + diurnal + noise


def generate_ecg_profile(
    date_obj: date,
    base_profile: np.ndarray = ECG_BASE_PROFILE,
) -> tuple:
    """Generate one day of ECG hourly demand with realistic variation."""
    month = date_obj.month
    dow = date_obj.weekday()
    iso = date_obj.isoformat()

    seasonal = SEASONAL_FACTOR[month]
    dow_mult = DOW_FACTOR[dow]
    is_hol = 1 if iso in HOLIDAYS else 0
    hol_factor = 0.90 if is_hol else 1.0

    profile = base_profile * seasonal * dow_mult * hol_factor

    temps = np.array([generate_ghana_temp(date_obj, h) for h in range(1, 25)])
    mean_temp = temps.mean()
    temp_deviation = mean_temp - 28.0
    temp_mult = 1.0 + temp_deviation * TEMP_SENSITIVITY
    profile = profile * temp_mult

    noise = np.random.normal(1.0, 0.02, size=24)
    profile = profile * noise

    profile = np.maximum(profile, 800.0)

    return profile, temps


async def seed():
    start_date = date(2025, 5, 1)
    end_date = date(2026, 5, 22)
    total_days = (end_date - start_date).days + 1

    logger.info(f"Seeding ECG historical data: {start_date} to {end_date} ({total_days} days)")

    async with AsyncSessionLocal() as db:
        existing = await db.execute(select(EcgHistoricalDemand).limit(1))
        if existing.scalar():
            logger.info("ECG historical data already seeded. Skipping.")
            print("Data already seeded. Drop table first if you want to re-seed.")
            return

        batch = []
        for i in range(total_days):
            d = start_date + timedelta(days=i)
            profile, temps = generate_ecg_profile(d)
            for h in range(24):
                batch.append({
                    "date": d,
                    "hour": h + 1,
                    "demand_mw": round(float(profile[h]), 2),
                    "temperature_c": round(float(temps[h]), 1),
                    "is_holiday": d.isoformat() in HOLIDAYS,
                })

        for rec in batch:
            db.add(EcgHistoricalDemand(**rec))
        await db.commit()

    logger.info(f"Seeded {total_days * 24} ECG hourly records.")
    print(f"DONE: {total_days} days ({total_days * 24} hourly records) seeded.")


if __name__ == "__main__":
    asyncio.run(seed())
