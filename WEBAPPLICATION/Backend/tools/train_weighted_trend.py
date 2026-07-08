"""Train WeightedTrendEngine from CSV data and save to models/"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
warnings = __import__('warnings')
warnings.filterwarnings('ignore')

DATA = r'C:\Users\User\Desktop\project\llm_from_scratch-main\LOADFORECASINGPROJECT\data\ecg_actual_demand_clean_with_temp.csv'
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'models', 'weighted_trend_engine.joblib')

df = pd.read_csv(DATA)
df['Date'] = pd.to_datetime(df['date'])
df['Hour'] = df['hour'].astype(int)
df['demand_mw'] = df['demand_mw'].astype(float)

from app.ml.weighted_trend_engine import WeightedTrendEngine
engine = WeightedTrendEngine()
engine.fit(df)
engine.save(OUT)
print(f"WeightedTrendEngine saved to {OUT}")
print(f"Profiles: {len(engine.profiles)} month×DOW combinations")
