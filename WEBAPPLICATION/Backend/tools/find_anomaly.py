
import pandas as pd

def find_drop():
    path = "../../../LOADFORECASINGPROJECT/resampled_data_15min.csv"
    print(f"Reading {path}...")
    df = pd.read_csv(path)
    
    # Check for DATETIME
    if 'DATETIME' in df.columns:
        df['timestamp'] = pd.to_datetime(df['DATETIME'])
    else:
        print("No DATETIME column")
        return

    df.sort_values('timestamp', inplace=True)
    
    t1 = "82T1_BANK (MW)"
    t3 = "82T3_BANK (MW)"
    t4 = "82T4_BANK (MW)"
    
    # Fill NaNs
    df[t1] = df[t1].fillna(0)
    df[t3] = df[t3].fillna(0)
    df[t4] = df[t4].fillna(0)
    
    df["load"] = df[t1] + df[t3] + df[t4]
    
    # Calculate rolling mean (24 hour = 96 steps)
    df["rolling_mean"] = df["load"].rolling(96).mean()
    
    # Find where rolling mean drops below threshold (e.g. 30 MW)
    # Start looking from the end backwards might be easier, or just find the first occurrence in the last chunk
    
    # Look at the last 5000 points
    tail = df.tail(5000).copy()
    
    # Find points where mean < 30
    low_load = tail[tail["rolling_mean"] < 30]
    
    if not low_load.empty:
        first_drop = low_load.iloc[0]
        print(f"Load drop detected!")
        print(f"First timestamp < 30 MW (Rolling 24h): {first_drop['timestamp']}")
        print(f"Value: {first_drop['load']}")
        print(f"Rolling Mean: {first_drop['rolling_mean']}")
        
        # Check a bit before
        idx = low_load.index[0]
        prior = df.loc[idx-100:idx]
        print("\nContext before drop:")
        print(prior[['timestamp', 'load', 'rolling_mean']].tail(10))
    else:
        print("No significant drop found in the last 5000 records.")

if __name__ == "__main__":
    find_drop()
