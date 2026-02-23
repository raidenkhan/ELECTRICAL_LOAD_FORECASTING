import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

def load_data(file_path):
    """Loads and preprocesses the dataset."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    df = pd.read_csv(file_path)
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])
    df.set_index('DATETIME', inplace=True)
    
    if 'DATE' in df.columns:
        df.drop(columns=['DATE'], inplace=True)
        
    return df

def plot_correlation_analysis(df, output_dir='plots'):
    """Generates ACF and PACF plots for load time series."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Identify Target Load Columns (MW)
    target_cols = [col for col in df.columns if '(MW)' in col]
    
    # Identify Load-Serving Transformer Columns (Community Load)
    # Based on EDA report: 
    # 82T3_BANK (MW) and 82T4_BANK (MW) are the main loads.
    # 82T1_BANK (MW) is auxiliary/light load.
    # 82T2_BANK (MW) has reverse flow (negative), likely generation, so we exclude it from "Demand" or treat separately.
    # AD2NY_LINE and NY6ZA_LINE are transmission lines, not loads.
    
    load_cols = ['82T3_BANK (MW)', '82T4_BANK (MW)', '82T1_BANK (MW)']
    
    # Verify columns exist before summing
    existing_load_cols = [col for col in load_cols if col in df.columns]
    
    if existing_load_cols:
        print(f"Calculating Community Load using: {existing_load_cols}")
        df['Community_Load_MW'] = df[existing_load_cols].sum(axis=1)
        
        # 1. Community Load ACF/PACF
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        series = df['Community_Load_MW'].dropna()
        
        plot_acf(series, lags=96*2, ax=ax1, title=f'Autocorrelation (ACF) - Community Load (Sum of {len(existing_load_cols)} Banks)')
        plot_pacf(series, lags=96*2, ax=ax2, title=f'Partial Autocorrelation (PACF) - Community Load')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'community_load_acf_pacf.png'))
        plt.close()
        print(f"Saved community_load_acf_pacf.png to {output_dir}")
    else:
        print("Warning: Load transformer columns not found. Skipping Community Load analysis.")

    # 2. Individual Line Analysis (Representative: AD2NY_LINE)
    # Check if column exists
    rep_col = 'AD2NY_LINE (MW)'
    if rep_col in df.columns:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        series_rep = df[rep_col].dropna()
        
        plot_acf(series_rep, lags=96*2, ax=ax1, title=f'Autocorrelation (ACF) - {rep_col}')
        plot_pacf(series_rep, lags=96*2, ax=ax2, title=f'Partial Autocorrelation (PACF) - {rep_col}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ad2ny_line_acf_pacf.png'))
        plt.close()
        print(f"Saved ad2ny_line_acf_pacf.png to {output_dir}")
    

from statsmodels.tsa.stattools import adfuller

def check_stationarity(series):
    # Copied from https://machinelearningmastery.com/time-series-data-stationary-python/

    result = adfuller(series.values)

    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

    if (result[1] <= 0.05) & (result[4]['5%'] > result[0]):
        print("\u001b[32mStationary\u001b[0m")
    else:
        print("\x1b[31mNon-stationary\x1b[0m")

if __name__ == "__main__":
    file_path = 'resampled_data_15min.csv'
    try:
        print("Loading data...")
        df = load_data(file_path)
        print("Data loaded. Performing correlation analysis...")
        plot_correlation_analysis(df)
        print("Analysis complete.")
        
        # Check stationarity
        if 'Community_Load_MW' in df.columns:
            check_stationarity(df['Community_Load_MW'])
        else:
            print("Community_Load_MW not calculated, skipping stationarity check.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
