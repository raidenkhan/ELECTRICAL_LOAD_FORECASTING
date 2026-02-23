
import pandas as pd
import numpy as np

def calculate_stats():
    path = "../../../LOADFORECASINGPROJECT/resampled_data_15min.csv"
    print(f"Reading {path}...")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Could not read CSV: {e}")
        return

    # Columns
    t1 = "82T1_BANK (MW)"
    t3 = "82T3_BANK (MW)"
    t4 = "82T4_BANK (MW)"
    
    # Calculate Total Load
    # Fill NaNs
    df[t1] = df[t1].fillna(0)
    df[t3] = df[t3].fillna(0)
    df[t4] = df[t4].fillna(0)
    
    df["Community_Load_MW"] = df[t1] + df[t3] + df[t4]
    
    # Calculate global stats
    mean_val = df["Community_Load_MW"].mean()
    std_val = df["Community_Load_MW"].std()
    
    print("-" * 50)
    print(f"GLOBAL STATS (N={len(df)})")
    print(f"Mean: {mean_val:.4f}")
    print(f"Std:  {std_val:.4f}")
    print("-" * 50)
    
    # Calculate last 3 months stats (approx 24*4*90 = 8640 rows)
    tail_df = df.tail(8640)
    tail_mean = tail_df["Community_Load_MW"].mean()
    tail_std = tail_df["Community_Load_MW"].std()
    
    print(f"LAST 3 MONTHS STATS (N={len(tail_df)})")
    print(f"Mean: {tail_mean:.4f}")
    print(f"Std:  {tail_std:.4f}")
    print("-" * 50)
    
    # Calculate last week
    week_df = df.tail(672)
    week_mean = week_df["Community_Load_MW"].mean()
    print(f"LAST WEEK MEAN: {week_mean:.4f}")

if __name__ == "__main__":
    calculate_stats()
