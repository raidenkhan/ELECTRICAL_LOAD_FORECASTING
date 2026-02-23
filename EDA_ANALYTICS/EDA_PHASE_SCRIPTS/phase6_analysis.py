import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
output_dir = 'plots/phase6'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])
    df.set_index('DATETIME', inplace=True)
    if 'DATE' in df.columns:
        df.drop(columns=['DATE'], inplace=True)
    return df

def analyze_target_redefinition(df, target_col):
    print("\n--- 13. Re-evaluate what you are forecasting ---")
    
    # 1. Raw Value vs Changes (Differencing)
    # Compare "Predictability" (Autocorrelation decay) of Raw vs Diff
    raw_corr = df[target_col].autocorr(lag=1)
    diff_corr = df[target_col].diff().autocorr(lag=1)
    
    print(f"Raw Series Autocorrelation (Lag 1): {raw_corr:.3f}")
    print(f"Differenced Series Autocorrelation (Lag 1): {diff_corr:.3f}")
    # High diff corr means "trend" is predictable. Low means random walk.
    
    # 2. Envelope Prediction (Daily Max/Min)
    # Is it easier to predict the Daily Peak than the 15-min value?
    daily_max = df[target_col].resample('D').max()
    daily_min = df[target_col].resample('D').min()
    
    # Naive Prediction for Envelope (Yesterday's Max predicts Today's Max)
    mae_max = (daily_max - daily_max.shift(1)).abs().mean()
    mae_min = (daily_min - daily_min.shift(1)).abs().mean()
    
    # For comparison, what is the daily average MAE of the 15-min persistence?
    # (Calculated in Phase 5: 11.79 MW).
    
    print(f"Daily Max Persistence MAE: {mae_max:.2f} MW")
    print(f"Daily Min Persistence MAE: {mae_min:.2f} MW")
    
    # Plot Envelopes
    plt.figure(figsize=(15, 6))
    daily_max.plot(label='Daily Max', color='red', alpha=0.6)
    daily_min.plot(label='Daily Min', color='blue', alpha=0.6)
    df[target_col].plot(label='Raw 15-min', color='grey', alpha=0.2)
    plt.title('Target Envelopes (Daily Min/Max) vs Raw Signal')
    plt.ylabel('MW')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'target_envelopes.png'))
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
            
            analyze_target_redefinition(df, target)
            
            print("\nPhase 6 Analysis Complete.")
        else:
            print("Error: Could not calculate Community Load.")
            
    except Exception as e:
        print(f"Error: {e}")
