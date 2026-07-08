"""Generate synthetic SCADA recent-week CSV for handover demo."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from app.ml.weighted_trend_engine import WeightedTrendEngine
import pandas as pd, numpy as np
from datetime import date, timedelta

engine = WeightedTrendEngine()
engine.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'models', 'weighted_trend_engine.joblib'))

rows = []
rng = np.random.default_rng(42)
base_level = 3600

for day_offset in range(7):
    d = date(2026, 5, 23) + timedelta(days=day_offset)
    dow = d.weekday()
    level = base_level + {0: 54, 1: 24, 2: 0, 3: 0, 4: 0, 5: -38, 6: -41}[dow]
    prof = engine.profiles.get((5, dow), np.ones(24))
    for h in range(24):
        hourly = prof[h] * level
        noise = hourly * 0.02 * rng.normal(0, 1)
        demand = max(0, round(hourly + noise, 1))
        rows.append({
            'date': d.isoformat(),
            'hour': h + 1,
            'demand_mw': demand,
            'temp_c': round(28 + rng.normal(0, 3), 1),
        })
    dm = np.mean([r['demand_mw'] for r in rows[-24:]])
    print(f'{d} DOW={dow}: level={level:.0f} mean={dm:.0f} MW')

df = pd.DataFrame(rows)
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'data', 'scada_recent_week.csv')
os.makedirs(os.path.dirname(out), exist_ok=True)
df.to_csv(out, index=False)
print(f'\nSaved {len(df)} rows to {out}')
print(f'Date range: {df.date.min()} to {df.date.max()}')
