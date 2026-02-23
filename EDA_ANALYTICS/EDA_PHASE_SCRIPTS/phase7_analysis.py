import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
output_dir = 'plots/phase7'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])
    df.set_index('DATETIME', inplace=True)
    if 'DATE' in df.columns:
        df.drop(columns=['DATE'], inplace=True)
    return df

def check_feature_leakage(df):
    print("\n--- 15. Feature Leakage Audit ---")
    
    # Common leakage source: Using "Future" information in Lag features accidentally
    # or using "Daily Average" (calculated from full day including future hours) as a feature.
    # We will simulate a standard feature engineering step and check timestamps.
    
    print("Checking temporal consistency...")
    
    # Mock Feature Engineering
    df_feat = df.copy()
    
    # Lag 1
    df_feat['Lag_1'] = df_feat['Community_Load_MW'].shift(1)
    
    # Rolling Mean (Center=False is CRITICAL)
    # If center=True, window looks ahead.
    df_feat['Rolling_Mean_24h_Correct'] = df_feat['Community_Load_MW'].rolling(window=96, center=False).mean()
    df_feat['Rolling_Mean_24h_Leaky'] = df_feat['Community_Load_MW'].rolling(window=96, center=True).mean()
    
    # Check correlation with Target
    target = df_feat['Community_Load_MW']
    
    corr_correct = target.corr(df_feat['Rolling_Mean_24h_Correct'])
    # Leaky correlation should be unnaturally high or different
    corr_leaky = target.corr(df_feat['Rolling_Mean_24h_Leaky'])
    
    print(f"Correlation (Correct Rolling Mean): {corr_correct:.3f}")
    print(f"Correlation (Leaky Center=True Mean): {corr_leaky:.3f}")
    
    if abs(corr_leaky) > abs(corr_correct) + 0.1:
        print("WARNING: Leaky feature shows significantly higher correlation. Ensure 'center=False' in all rolling windows.")
    else:
        print("Leakage check passed for Rolling Mean example.")

def define_benchmarks(df, target_col):
    print("\n--- 14. Baseline Benchmarks ---")
    
    # Metrics: MAE, RMSE, MAPE
    
    # 1. Persistence (t-96) - Seasonal Naive
    y_true = df[target_col]
    y_pred_seasonal = df[target_col].shift(96)
    
    mask = ~y_true.isna() & ~y_pred_seasonal.isna()
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred_seasonal[mask]
    
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    mape = np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean)) * 100
    
    print(f"Benchmark: Seasonal Persistence (24h)")
    print(f"MAE: {mae:.2f} MW")
    print(f"RMSE: {rmse:.2f} MW")
    print(f"MAPE: {mape:.2f}%")
    
    # Plot Evaluation
    plt.figure(figsize=(15, 6))
    # Plot last 7 days only for clarity
    days_to_plot = 7
    subset = -96 * days_to_plot
    
    plt.plot(y_true_clean.index[subset:], y_true_clean.values[subset:], label='Actual', color='black')
    plt.plot(y_true_clean.index[subset:], y_pred_clean.values[subset:], label='Benchmark (Seasonal)', color='red', linestyle='--')
    plt.title(f'Benchmark Performance (Last {days_to_plot} Days)')
    plt.ylabel('MW')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'benchmark_comparison.png'))
    plt.close()

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
            
            check_feature_leakage(df)
            define_benchmarks(df, target)
            
            print("\nPhase 7 Analysis Complete.")
        else:
            print("Error: Could not calculate Community Load.")
            
    except Exception as e:
        print(f"Error: {e}")
