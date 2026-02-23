import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
output_dir = 'plots/phase5'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])
    df.set_index('DATETIME', inplace=True)
    if 'DATE' in df.columns:
        df.drop(columns=['DATE'], inplace=True)
    return df

def analyze_horizon_predictability(df, target_col):
    print("\n--- 11. Horizon-Dependent Predictability ---")
    
    # Simple Naive Forecast: Persistence (t-96 for 24h seasonality, or t-1 for immediate)
    # We will test "Persistence 24h" (Forecast(t) = Actual(t-96)) as the baseline.
    # We examine MAE vs Horizon (Steps ahead: 1 to 96)
    
    # Just checking error of using t-k as predictor for t, across varying k (horizon)
    # Ideally: Predict t+h given t.
    # Naive approach: Pred(t+h) = Actual(t).
    
    horizons = [1, 4, 8, 12, 24, 48, 96] # 15m, 1h, 2h, 3h, 6h, 12h, 24h
    maes = []
    
    # Train/Test split for evaluation (Last 30 days)
    # Actually just rolling eval over whole dataset for EDA
    
    for h in horizons:
        # Naive Forecast: Pred(t+h) = Actual(t)
        # Shift target by h to align: df[t] vs df[t-h]
        # Pred = df[target_col].shift(h)
        # Truth = df[target_col]
        pred = df[target_col].shift(h)
        mae = (df[target_col] - pred).abs().mean()
        maes.append(mae)
        
    plt.figure(figsize=(10, 6))
    plt.plot(horizons, maes, marker='o', linestyle='-', linewidth=2)
    plt.title(f'Naive Forecast Error (MAE) vs Planning Horizon')
    plt.xlabel('Horizon (steps of 15 min)')
    plt.ylabel('MAE (MW)')
    plt.grid(True)
    plt.xticks(horizons)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'horizon_predictability.png'))
    plt.close()
    
    print("Baseline MAE (Persistence from t-h):")
    for h, mae in zip(horizons, maes):
        print(f"h={h} ({h*15}m): {mae:.2f} MW")
        
    # Also check "Seasonal Persistence" (t-96)
    # For h=1..96, we usually predict t+h. If we use daily persistence: Pred(t+h) = Actual(t+h-96).
    # That is constant error = MAE(t, t-96). 
    # Let's calculate that specific baseline.
    mae_seasonal = (df[target_col] - df[target_col].shift(96)).abs().mean()
    print(f"Seasonal Persistence (24h ago): {mae_seasonal:.2f} MW")

def analyze_variance_target(df, target_col):
    print("\n--- 12. Variance as a Target ---")
    
    # 1. Rolling Volatility (Standard Deviation of window)
    window_size = 4 * 24 # 1 day rolling window
    rolling_std = df[target_col].rolling(window=window_size).std()
    
    plt.figure(figsize=(15, 6))
    rolling_std.plot(color='orange', linewidth=1)
    plt.title(f'Rolling 24h Volatility (Standard Deviation)')
    plt.ylabel('MW Deviation')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rolling_volatility.png'))
    plt.close()
    
    # 2. Volatility by Hour of Day
    df_feat = df[[target_col]].copy()
    df_feat['Hour'] = df_feat.index.hour
    
    # Calculate std dev of load for each hour across all days
    hourly_volatility = df_feat.groupby('Hour')[target_col].std()
    
    plt.figure(figsize=(10, 6))
    hourly_volatility.plot(kind='bar', color='purple', alpha=0.7)
    plt.title('Load Volatility by Hour of Day (Std Dev)')
    plt.ylabel('MW Std Dev')
    plt.xlabel('Hour (0-23)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hourly_volatility.png'))
    plt.close()
    
    print("Top 3 Most Volatile Hours:")
    print(hourly_volatility.sort_values(ascending=False).head(3).to_string())

if __name__ == "__main__":
    file_path = 'resampled_data_15min.csv'
    try:
        df = load_data(file_path)
        
        # Calculate Community Load
        load_cols = ['82T3_BANK (MW)', '82T4_BANK (MW)', '82T1_BANK (MW)']
        existing_cols = [c for c in load_cols if c in df.columns]
        
        if existing_cols:
            df['Community_Load_MW'] = df[existing_cols].sum(axis=1)
            target = 'Community_Load_MW'
            print(f"Analyzing Target: {target}")
            
            analyze_horizon_predictability(df, target)
            analyze_variance_target(df, target)
            
            print("\nPhase 5 Analysis Complete.")
        else:
            print("Error: Could not calculate Community Load.")
            
    except Exception as e:
        print(f"Error: {e}")
