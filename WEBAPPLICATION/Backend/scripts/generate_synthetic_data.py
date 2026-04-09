import pandas as pd
import numpy as np
import datetime
import os

def generate_community_load_dataset(output_path, start_date='2025-01-01', end_date='2025-12-31'):
    """
    Generates a massive synthetic dataset mimicking community load patterns (~80 MW).
    """
    print(f"Generating synthetic data from {start_date} to {end_date}...")
    
    # 1. Create time index (15-minute intervals)
    timestamps = pd.date_range(start=start_date, end=end_date, freq='15min')
    n_points = len(timestamps)
    
    # 2. Base Load Logic (Centered around 80 MW)
    base_load = 80.0
    
    # 3. Daily Profile (Double Peak: Morning 08:00, Evening 19:00)
    # Use a mixture of Gaussians or Sine waves for the daily cycle
    hour = timestamps.hour + timestamps.minute / 60.0
    daily_cycle = (
        12.0 * np.exp(-((hour - 8.5)**2) / 4.0) +  # Morning peak
        18.0 * np.exp(-((hour - 19.5)**2) / 6.0) + # Evening peak
        -5.0 * np.exp(-((hour - 3.0)**2) / 2.0)    # Night trough
    )
    
    # 4. Weekly Pattern (Weekends -10%)
    is_weekend = timestamps.weekday >= 5
    weekly_multiplier = np.where(is_weekend, 0.90, 1.0)
    
    # 5. Seasonal Cycle (Summer/Winter peaks)
    # Day of year sine wave
    doy = timestamps.dayofyear
    seasonal_cycle = 10.0 * np.sin(2 * np.pi * (doy - 100) / 365) + 5.0 * np.cos(4 * np.pi * (doy - 10) / 365)
    
    # 6. Combine Components
    load = (base_load + daily_cycle + seasonal_cycle) * weekly_multiplier
    
    # 7. Add Gaussian Noise
    noise = np.random.normal(0, 2.5, n_points)
    load += noise
    
    # 8. Secondary Features
    # Voltage: 33kV +/- 0.5kV
    voltage = 33.0 + np.random.normal(0, 0.2, n_points) + 0.3 * np.sin(2 * np.pi * hour / 24)
    
    # Frequency: 50Hz +/- 0.1Hz
    frequency = 50.0 + np.random.normal(0, 0.02, n_points)
    
    # Temperature: 15-35C
    temp = 22.0 + 8.0 * np.sin(2 * np.pi * (doy - 120) / 365) + 4.0 * np.sin(2 * np.pi * (hour - 14) / 24)
    
    # Current (Approx P = sqrt(3) * V * I * pf, assume pf=0.9 and balanced)
    # I = P / (sqrt(3) * V * pf)
    current = (load * 1000) / (np.sqrt(3) * voltage * 0.9)
    
    # 9. Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps.strftime('%Y-%m-%d %H:%M:%S'),
        'total_load_mw': np.round(load, 3),
        'voltage_kv': np.round(voltage, 3),
        'current_a': np.round(current, 2),
        'frequency_hz': np.round(frequency, 3),
        'temperature_c': np.round(temp, 2)
    })
    
    # 10. Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Successfully generated {len(df)} rows at {output_path}")
    print(f"Mean Load: {df['total_load_mw'].mean():.2f} MW")
    print(f"Peak Load: {df['total_load_mw'].max():.2f} MW")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir))) # Adjust based on structure
    output_file = os.path.join(script_dir, "massive_synthetic_load.csv")
    
    generate_community_load_dataset(output_file)
