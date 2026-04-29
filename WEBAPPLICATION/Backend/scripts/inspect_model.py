import joblib
import os
import numpy as np

model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/decomp_engine.joblib'))
if os.path.exists(model_path):
    state = joblib.load(model_path)
    trend = state['trend']
    seasonal = state['seasonal']
    
    print(f"Trend last_val: {trend.last_val}")
    print(f"Seasonal s_ts mean: {seasonal.s_ts.mean()}")
    print(f"Seasonal s_ts first 5: {seasonal.s_ts[:5]}")
    print(f"Seasonal s_dow: {seasonal.s_dow}")
else:
    print("Model file not found")
