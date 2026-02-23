import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
output_dir = 'plots/phase4'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])
    df.set_index('DATETIME', inplace=True)
    if 'DATE' in df.columns:
        df.drop(columns=['DATE'], inplace=True)
    return df

def analyze_cross_correlation_leads(df, target_col, all_mw_cols):
    print("\n--- 9. Cross-Correlation and Lead-Lag Analysis ---")
    
    # We want to see if any line 'leads' the target (Community Load)
    # i.e., correlation(line(t-lag), target(t))
    
    lags = [-8, -4, -2, -1, 0, 1, 2, 4, 8] # Negative lag = Line Leads Target? 
    # Usually: corr(x(t-k), y(t)). If peak is at k>0, x leads y.
    
    results = {}
    
    # Prepare data (fill NA for calculation)
    df_clean = df.fillna(method='ffill').fillna(method='bfill')
    
    for col in all_mw_cols:
        if col == target_col: continue
        
        corrs = []
        for lag in lags:
            # Shift col by lag
            # IF lag is positive (e.g., 1), we are looking at col(t-1) vs target(t). 
            # If correlation is high here, past col predicts current target.
            shifted = df_clean[col].shift(lag)
            corr = df_clean[target_col].corr(shifted)
            corrs.append(corr)
            
        results[col] = corrs
        
    # Plot Lead-Lag
    plt.figure(figsize=(12, 8))
    for col, corrs in results.items():
        plt.plot(lags, corrs, marker='o', label=col)
        
    plt.title(f'Lead-Lag Cross-Correlation with {target_col}')
    plt.xlabel('Lag (15-min intervals) [Positive Lag = Column Leading/Predictive]')
    plt.ylabel('Correlation')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cross_correlation_lag.png'))
    plt.close()
    
    # Print strongest predictors (Max absolute correlation at positive lags)
    print("Max Predictive Correlation (Lags > 0):")
    for col, corrs in results.items():
        # Lags indices: 0(-8), 1(-4), ... 4(0), 5(1), 6(2), 7(4), 8(8)
        # We care about indices 5, 6, 7, 8 (Positive Lags)
        pos_lag_corrs = corrs[5:]
        max_corr = max(pos_lag_corrs, key=abs)
        print(f"{col}: {max_corr:.3f}")

def analyze_redundancy(df, all_mw_cols):
    print("\n--- 10. Redundancy Analysis ---")
    
    # 1. Correlation Matrix
    corr_matrix = df[all_mw_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title('Correlation Matrix (All Lines)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'redundancy_corr_matrix.png'))
    plt.close()
    
    # 2. PCA to estimate "True" dimensionality
    df_clean = df[all_mw_cols].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)
    
    pca = PCA()
    pca.fit(X_scaled)
    
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    print("\nPCA Explained Variance:")
    for i, (ev, cum) in enumerate(zip(explained_variance, cumulative_variance)):
        print(f"PC{i+1}: {ev:.4f} (Cumulative: {cum:.4f})")
        
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(explained_variance)+1), cumulative_variance, marker='o', linestyle='--')
    plt.axhline(y=0.95, color='r', linestyle=':', label='95% Variance')
    plt.title('PCA Cumulative Explained Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Variance Ratio')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'redundancy_pca.png'))
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
            
            # Analyze all MW columns + Computed Load
            mw_cols = [col for col in df.columns if '(MW)' in col]
            
            analyze_cross_correlation_leads(df, 'Community_Load_MW', mw_cols)
            analyze_redundancy(df, mw_cols)
            
            print("\nPhase 4 Analysis Complete.")
        else:
            print("Error: Could not calculate Community Load.")
            
    except Exception as e:
        print(f"Error: {e}")
