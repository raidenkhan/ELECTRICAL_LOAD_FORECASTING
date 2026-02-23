import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('resampled_data_15min.csv')
target_cols = [c for c in df.columns if '(MW)' in c]
kv_cols = [c for c in df.columns if '(KV)' in c]
mx_cols = [c for c in df.columns if '(MX)' in c]
temp_cols = [c for c in df.columns if 'TEMPERATURE' in c]

# 1. Temperature vs Load Analysis
print("--- Temperature Analysis ---")
if len(temp_cols) > 0 and '82T3_BANK (MW)' in df.columns:
    temp_col = temp_cols[0]
    corr = df[temp_col].corr(df['82T3_BANK (MW)'])
    print(f"Correlation between {temp_col} and 82T3_BANK (MW): {corr:.4f}")
    # Correlation with all loads
    print("Correlation Table (Temp vs Loads):")
    print(df[[temp_col] + target_cols[:3]].corr().iloc[0])

# 2. Voltage Stability
print("\n--- Voltage Analysis ---")
print(df[kv_cols].describe().T[['mean', 'std', 'min', 'max']])

# 3. Reactive Power
print("\n--- Reactive Power Analysis ---")
print(df[mx_cols].describe().T[['mean', 'std', 'min', 'max']])

# 4. Frequency
print("\n--- Frequency Analysis ---")
if 'FREQ (HZ)' in df.columns:
    print(df['FREQ (HZ)'].describe())
