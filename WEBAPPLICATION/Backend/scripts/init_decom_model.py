import sys
import os
import pandas as pd
import numpy as np

# Add the project root to sys.path to allow imports from app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.ml.decom_engine import DecomEngine
from app.core.config import settings

def init_model():
    print("=" * 60)
    print("INITIALIZING DECOMPOSITION MODEL")
    print("=" * 60)
    
    # 1. Load Data (same as experimental script)
    scada_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../EXTRAS/resampled_data_15min.csv'))
    meteo_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../openmeteoweather.csv'))
    
    if not os.path.exists(scada_file):
        print(f"Error: SCADA file not found at {scada_file}")
        return

    df = pd.read_csv(scada_file)
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])
    df = df.sort_values('DATETIME').reset_index(drop=True)

    # Clean target
    t1 = df['82T1_BANK (MW)'].clip(lower=0)
    t3 = df['82T3_BANK (MW)']
    t4 = df['82T4_BANK (MW)']
    df['Load'] = (t1 + t3 + t4).clip(lower=0)
    
    # Community Load Outage Threshold (25 MW)
    # Values below 25 MW are considered grid outages/sensor failure and should not bias the model
    df['Load'][df['Load'] < 25.0] = np.nan
    df['Load'] = df['Load'].interpolate(method='pchip').ffill().bfill()

    # Weather
    try:
        meteo = pd.read_csv(meteo_file, skiprows=3)
        meteo['DATETIME'] = pd.to_datetime(meteo['time'])
        meteo = (meteo.set_index('DATETIME')[['temperature_2m (°C)']]
                     .resample('15T').interpolate(method='linear').reset_index())
        meteo.rename(columns={'temperature_2m (°C)': 'Temp'}, inplace=True)
        df = pd.merge(df, meteo, on='DATETIME', how='left')
        df['Temp'] = df['Temp'].interpolate(method='linear').ffill().fillna(28.0)
    except:
        df['Temp'] = 28.0

    # Features
    df['Date']       = df['DATETIME'].dt.date
    df['TimeSlot']   = df['DATETIME'].dt.hour * 4 + df['DATETIME'].dt.minute // 15
    df['DOW']        = df['DATETIME'].dt.dayofweek
    df['Is_Holiday'] = 0 # Baseline without holidays for initial load
    
    # Outage Masking
    mu, sig = df['Load'].mean(), df['Load'].std()
    thresh = mu - 2.0 * sig
    df['Is_Outage'] = (df.groupby('Date')['Load'].transform('mean') < thresh).astype(int)
    df['Masked_Load'] = df['Load'].copy()
    df.loc[df['Is_Outage'] == 1, 'Masked_Load'] = np.nan
    df['Masked_Load'] = df['Masked_Load'].interpolate(method='pchip').ffill().bfill()

    # 2. Fit Engine
    engine = DecomEngine()
    print("Fitting components...")
    
    # Trend
    daily_mean = df[df['Is_Outage'] == 0].groupby('Date')['Masked_Load'].mean().dropna()
    engine.trend.fit(daily_mean)
    
    # Seasonal
    engine.seasonal.fit(df[df['Is_Outage'] == 0])
    
    # Temperature (Linearize Trend x Seasonal first)
    trend_arr = engine.trend.get_trend_array(df['Date'])
    s_arr = engine.seasonal.apply(df['TimeSlot'].values, df['DOW'].values)
    ratio_ts = df['Masked_Load'] / (trend_arr * s_arr)
    engine.temp.fit(df['Temp'].values, ratio_ts)
    
    # Holiday (Placeholder fit)
    engine.holiday.fit(df['TimeSlot'].values, df['Is_Holiday'].values, ratio_ts)
    
    engine.is_fitted = True
    
    # 3. Save
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/decomp_engine.joblib'))
    engine.save(model_path)
    print(f"Success! Initialization complete. Model saved to {model_path}")

if __name__ == "__main__":
    init_model()
