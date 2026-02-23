
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import os
import sys

# Add parent directory to path to import config if needed, or just hardcode for the tool
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def train_and_save_ltlf():
    print("Training LTLF Recursive Model...")
    
    # Path settings
    DATA_PATH = "../../resampled_data_15min.csv" # Relative to Backend/tools
    MODEL_PATH = "models/ltlf_recursive.pkl"
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {os.path.abspath(DATA_PATH)}")
        return
        
    # 1. Load Data
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH, index_col='DATETIME', parse_dates=True)
    df['Community_Load_MW'] = df['82T1_BANK (MW)'] + df['82T3_BANK (MW)'] + df['82T4_BANK (MW)']
    
    # 2. Aggregate to Daily Peak
    df_daily = df['Community_Load_MW'].resample('D').max().to_frame()
    df_daily.columns = ['Peak_MW']
    df_daily = df_daily[df_daily['Peak_MW'] > 50] # Filter outages
    
    # 3. Create Features
    print("Creating features...")
    df_daily['DayOfWeek'] = df_daily.index.dayofweek
    df_daily['Month'] = df_daily.index.month
    df_daily['DayOfYear'] = df_daily.index.dayofyear
    df_daily['Lag_1'] = df_daily['Peak_MW'].shift(1)
    df_daily['Lag_7'] = df_daily['Peak_MW'].shift(7)
    
    df_daily = df_daily.dropna()
    
    features = ['DayOfWeek', 'Month', 'DayOfYear', 'Lag_1', 'Lag_7']
    target = 'Peak_MW'
    
    # 4. Train Models (P10, P50, P90)
    models = {}
    alphas = [0.1, 0.5, 0.9]
    
    for alpha in alphas:
        print(f"Training Quantile {alpha}...")
        model = lgb.LGBMRegressor(
            objective='quantile',
            alpha=alpha,
            n_estimators=500, # Reduced slightly for speed, sufficient for daily data
            learning_rate=0.01,
            max_depth=5,
            num_leaves=20,
            random_state=42,
            verbose=-1
        )
        model.fit(df_daily[features], df_daily[target])
        models[alpha] = model
        
    # 5. Save metadata and models
    artifact = {
        "models": models,
        "features": features,
        "last_train_date": df_daily.index[-1],
        "last_peak_history": list(df_daily['Peak_MW'].tail(7).values) # Needed for recursion
    }
    
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    print(f"Successfully saved LTLF model artifact to {MODEL_PATH}")

if __name__ == "__main__":
    train_and_save_ltlf()
