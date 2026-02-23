
import pandas as pd

def calc_valid_stats():
    path = "../../../LOADFORECASINGPROJECT/resampled_data_15min.csv"
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Error: {e}")
        return

    if 'DATETIME' in df.columns:
        df['timestamp'] = pd.to_datetime(df['DATETIME'])
    else:
        print("No DATETIME column")
        return
        
    # Filter valid data
    cutoff = '2025-05-01'
    df = df[df['timestamp'] < cutoff]
    
    t1 = "82T1_BANK (MW)"
    t3 = "82T3_BANK (MW)"
    t4 = "82T4_BANK (MW)"
    
    df[t1] = df[t1].fillna(0)
    df[t3] = df[t3].fillna(0)
    df[t4] = df[t4].fillna(0)
    
    df["load"] = df[t1] + df[t3] + df[t4]
    
    mean_val = df["load"].mean()
    std_val = df["load"].std()
    
    print(f"Stats for data before {cutoff} (N={len(df)})")
    print(f"Mean: {mean_val:.4f}")
    print(f"Std:  {std_val:.4f}")

if __name__ == "__main__":
    calc_valid_stats()
