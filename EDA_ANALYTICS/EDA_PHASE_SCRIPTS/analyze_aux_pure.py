import csv
import math

def calculate_mean(data):
    if not data: return 0
    return sum(data) / len(data)

def calculate_std_dev(data, mean):
    if not data or len(data) < 2: return 0
    variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
    return math.sqrt(variance)

def calculate_correlation(x, y):
    if not x or not y or len(x) != len(y): return 0
    n = len(x)
    mu_x = calculate_mean(x)
    mu_y = calculate_mean(y)
    numerator = sum((xi - mu_x) * (yi - mu_y) for xi, yi in zip(x, y))
    sum_sq_diff_x = sum((xi - mu_x) ** 2 for xi in x)
    sum_sq_diff_y = sum((yi - mu_y) ** 2 for yi in y)
    denominator = math.sqrt(sum_sq_diff_x * sum_sq_diff_y)
    return numerator / denominator if denominator != 0 else 0

file_path = 'resampled_data_15min.csv'

# Indices (based on inspection or header reading)
# Will read header first to be dynamic
target_mw_col = '82T3_BANK (MW)'
kv_keyword = '(KV)'
temp_keyword = 'TEMPERATURE'
mx_keyword = '(MX)'

data_mw = []
data_temp = []
data_kv = {} # dict of list
data_mx = {}

try:
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        # Handle the second row unit/description if present or just data
        # Based on snippet, first row is headers. 
        # Wait, snippet showed:
        # 122:     <tr>
        # 123:       <th>DATETIME</th>
        # This implies it might be a pandas dataframe display, not CSV content.
        # But `head` output earlier showed: DATETIME,DATE,AD2NY_LINE (A)...
        # So it's a standard CSV header.
        
        mw_idx = -1
        temp_idx = -1
        kv_indices = []
        mx_indices = []
        
        for i, h in enumerate(headers):
            if h == target_mw_col:
                mw_idx = i
            elif temp_keyword in h and temp_idx == -1: # Take first temp
                temp_idx = i
                print(f"Using {h} as Temperature proxy")
            
            if kv_keyword in h:
                kv_indices.append(i)
                data_kv[h] = []
            
            if mx_keyword in h:
                mx_indices.append(i)
                data_mx[h] = []

        row_count = 0
        for row in reader:
            if not row: continue
            try:
                # MW
                if mw_idx != -1:
                    val = float(row[mw_idx])
                    data_mw.append(val)
                
                # Temp
                if temp_idx != -1:
                    val = float(row[temp_idx])
                    data_temp.append(val)
                
                # KV
                for idx in kv_indices:
                    val = float(row[idx])
                    data_kv[headers[idx]].append(val)
                    
                # MX
                for idx in mx_indices:
                    val = float(row[idx])
                    data_mx[headers[idx]].append(val)
                
                row_count += 1
            except ValueError:
                continue

    print(f"\nProcessed {row_count} rows.")
    
    # 1. Temp vs Load
    if data_mw and data_temp:
        corr = calculate_correlation(data_temp, data_mw)
        print(f"Correlation (Temp vs Load): {corr:.4f}")
    
    # 2. Voltage Stats
    print("\nVoltage (KV) Stats:")
    for col, values in data_kv.items():
        if not values: continue
        mu = calculate_mean(values)
        std = calculate_std_dev(values, mu)
        print(f"{col}: Mean={mu:.2f}, Std={std:.2f}, Min={min(values):.2f}, Max={max(values):.2f}")
        
    # 3. Reactive Power Stats
    print("\nReactive Power (MX) Stats (Sample):")
    count = 0
    for col, values in data_mx.items():
        if not values: continue
        if count >= 3: break 
        mu = calculate_mean(values)
        print(f"{col}: Mean={mu:.2f}")
        count += 1

except Exception as e:
    print(f"Error in pure python analysis: {e}")
