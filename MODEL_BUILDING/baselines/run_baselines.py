"""
Baseline Model Runner.

Implements and evaluates:
1. Persistence Model (Naive 1): Predicts t using t-1
2. Seasonal Naive Model (Naive 2): Predicts t using t-96 (24h ago)
3. Linear Regression: Basic OLS using engineered features

Author: Load Forecasting Team
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import joblib

# Add parent directory to path to import metrics
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics import calculate_metrics, print_metrics

# --- Configuration ---
DATA_PATH = r"c:\Users\User\Desktop\project\llm_from_scratch-main\LOADFORECASINGPROJECT\FEATURE_ENGINEERING\outputs\engineered_features.csv"
OUTPUT_DIR = r"c:\Users\User\Desktop\project\llm_from_scratch-main\LOADFORECASINGPROJECT\MODEL_BUILDING\results"
PLOT_DIR = r"c:\Users\User\Desktop\project\llm_from_scratch-main\LOADFORECASINGPROJECT\MODEL_BUILDING\plots"

# Date for Train/Test Split (Simulation of 'Live' deployment)
# Train: Jan 2024 -> April 2025
# Test: May 2025 -> June 2025
SPLIT_DATE = "2025-05-01"

TARGET_COL = "Community_Load_MW"

# Features for Linear Regression
FEATURES = [
    "Lag_1", "Lag_4", "Lag_96", "Lag_672",
    "Rolling_Mean_24h", "Rolling_Max_24h",
    "NY6ZA_Flow", "T2_Generation",
    "Hour_Sin", "Hour_Cos", "IsWeekend"
]

def load_data():
    """Load and prepare data."""
    print(f"Loading data from {DATA_PATH}...")
    
    # Read CSV. The first column is the index (Datetime) but has no name in the CSV header
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    
    # Ensure index is named Datetime
    df.index.name = 'Datetime'
    
    # Drop rows with NaN (from lags)
    df = df.dropna()
    print(f"Loaded data range: {df.index.min()} to {df.index.max()}")
    return df

def run_persistence(train, test):
    """
    Persistence Model: Forecast(t) = Actual(t-1)
    In our feature set, Lag_1 is exactly Actual(t-1)
    """
    print("\n--- Running Persistence Model ---")
    y_true = test[TARGET_COL]
    y_pred = test['Lag_1']
    
    metrics = calculate_metrics(y_true, y_pred)
    print_metrics(metrics, "Persistence (Lag 1)")
    return metrics, y_pred

def run_seasonal_naive(train, test):
    """
    Seasonal Naive Model: Forecast(t) = Actual(t-96)
    In our feature set, Lag_96 is exactly Actual(t-96) (24h ago)
    """
    print("\n--- Running Seasonal Naive Model ---")
    y_true = test[TARGET_COL]
    y_pred = test['Lag_96']
    
    metrics = calculate_metrics(y_true, y_pred)
    print_metrics(metrics, "Seasonal Naive (24h)")
    return metrics, y_pred

def run_linear_regression(train, test):
    """
    Linear Regression Model using engineered features.
    """
    print("\n--- Running Linear Regression ---")
    
    X_train = train[FEATURES]
    y_train = train[TARGET_COL]
    X_test = test[FEATURES]
    y_test = test[TARGET_COL]
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    metrics = calculate_metrics(y_test, y_pred)
    print_metrics(metrics, "Linear Regression")
    
    return metrics, y_pred, model

def plot_results(test_data, preds_dict):
    """Generate plots for visual verification."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    # 1. Time Series Plot (Zoomed in 1 week)
    plt.figure(figsize=(15, 7))
    
    # Select a 1-week window from test set
    start_idx = 0
    end_idx = 96 * 7 # 7 days
    
    subset_dates = test_data.index[start_idx:end_idx]
    y_true_subset = test_data[TARGET_COL].iloc[start_idx:end_idx]
    
    plt.plot(subset_dates, y_true_subset, label='Actual', color='black', linewidth=2)
    
    colors = ['red', 'green', 'blue']
    for i, (name, preds) in enumerate(preds_dict.items()):
        plt.plot(subset_dates, preds[start_idx:end_idx], label=name, color=colors[i], linestyle='--')
        
    plt.title('Baseline Models Forecast vs Actual (1 Week Zoom)')
    plt.ylabel('Load (MW)')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(PLOT_DIR, 'baselines_forecast.png'))
    plt.close()
    print(f"Plot saved to {os.path.join(PLOT_DIR, 'baselines_forecast.png')}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    df = load_data()
    
    # Chronological Split
    train = df[df.index < SPLIT_DATE]
    test = df[df.index >= SPLIT_DATE]
    
    print(f"Train samples: {len(train)}")
    print(f"Test samples: {len(test)}")
    
    # Store results
    results = []
    predictions = {}
    
    # 1. Persistence
    m1, p1 = run_persistence(train, test)
    results.append({'Model': 'Persistence', **m1})
    predictions['Persistence'] = p1
    
    # 2. Seasonal Naive
    m2, p2 = run_seasonal_naive(train, test)
    results.append({'Model': 'Seasonal Naive', **m2})
    predictions['Seasonal Naive'] = p2
    
    # 3. Linear Regression
    m3, p3, lr_model = run_linear_regression(train, test)
    results.append({'Model': 'Linear Regression', **m3})
    predictions['Linear Regression'] = p3
    
    # Save Results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'baseline_results.csv'), index=False)
    print(f"\nResults saved to {os.path.join(OUTPUT_DIR, 'baseline_results.csv')}")
    
    # Save LR model
    joblib.dump(lr_model, os.path.join(OUTPUT_DIR, 'linear_regression.pkl'))
    
    # Plotting
    plot_results(test, predictions)

if __name__ == "__main__":
    main()
