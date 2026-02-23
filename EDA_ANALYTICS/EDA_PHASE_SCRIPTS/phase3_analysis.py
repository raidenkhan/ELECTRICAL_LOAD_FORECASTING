import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
output_dir = 'plots/phase3'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])
    df.set_index('DATETIME', inplace=True)
    if 'DATE' in df.columns:
        df.drop(columns=['DATE'], inplace=True)
    return df

def analyze_regime_segmentation(df, target_col):
    print("\n--- 7. Regime Segmentation ---")
    
    # Create daily profiles
    # Pivot: Index=Date, Columns=Time_Index (0-95), Value=Load
    df_feat = df[[target_col]].copy()
    df_feat['Date'] = df_feat.index.date
    df_feat['Time_Index'] = df_feat.index.hour * 4 + df_feat.index.minute / 15
    
    daily_profiles = df_feat.pivot(index='Date', columns='Time_Index', values=target_col)
    daily_profiles.dropna(inplace=True) # Drop incomplete days
    
    # Scale for clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(daily_profiles)
    
    # K-Means Clustering
    # Try 3 clusters (e.g., High, Med, Low or Summer, Winter, Transition)
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    daily_profiles['Cluster'] = labels
    
    # Count days per cluster
    print("Days per Regime (Cluster):")
    print(daily_profiles['Cluster'].value_counts().sort_index())
    
    # Plot Average Profile per Cluster
    plt.figure(figsize=(12, 6))
    for i in range(n_clusters):
        cluster_data = daily_profiles[daily_profiles['Cluster'] == i].drop(columns='Cluster')
        mean_profile = cluster_data.mean()
        plt.plot(mean_profile.index, mean_profile.values, label=f'Regime {i} (n={len(cluster_data)})', linewidth=2)
        
    plt.title(f'Regime Segmentation: Average Daily Profiles ({target_col})')
    plt.ylabel('MW')
    plt.xlabel('15-min Interval Index')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'regime_profiles.png'))
    plt.close()
    
    # Visualize Clusters in PCA Space (2D)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.title('Regime Clusters (PCA Projection)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(*scatter.legend_elements(), title="Regime")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'regime_pca_clusters.png'))
    plt.close()

def analyze_event_fingerprinting(df, target_col):
    print("\n--- 8. Event Fingerprinting ---")
    
    # Identify "Abnormal" Days based on basic stats (e.g., Daily Max Load > 99th percentile or < 1st)
    daily_max = df[target_col].resample('D').max()
    threshold_high = daily_max.quantile(0.99)
    threshold_low = daily_max.quantile(0.01) # Low load days might be outages
    
    extreme_days_high = daily_max[daily_max > threshold_high].index
    extreme_days_low = daily_max[daily_max < threshold_low].index
    
    print(f"High Extreme Threshold: {threshold_high:.2f} MW")
    print(f"Number of High Extreme Days: {len(extreme_days_high)}")
    print(f"Low Extreme Threshold: {threshold_low:.2f} MW")
    print(f"Number of Low Extreme Days: {len(extreme_days_low)}")
    
    # Prepare profile data again
    df_feat = df[[target_col]].copy()
    df_feat['Date'] = df_feat.index.date
    df_feat['Time_Index'] = df_feat.index.hour * 4 + df_feat.index.minute / 15
    daily_profiles = df_feat.pivot(index='Date', columns='Time_Index', values=target_col)
    
    # Convert index to datetime for matching
    daily_profiles.index = pd.to_datetime(daily_profiles.index)
    
    # Plot High Extreme Days vs Normal Average
    plt.figure(figsize=(12, 6))
    
    # Plot background (Normal days - grey)
    normal_days_mask = ~daily_profiles.index.isin(extreme_days_high) & ~daily_profiles.index.isin(extreme_days_low)
    normal_avg = daily_profiles[normal_days_mask].mean()
    plt.plot(normal_avg.index, normal_avg.values, 'k--', linewidth=2, label='Normal Average')
    
    # Overlay High Extremes (Red)
    for date in extreme_days_high:
        if date in daily_profiles.index:
            plt.plot(daily_profiles.loc[date].index, daily_profiles.loc[date].values, 'r-', alpha=0.3)
            
    # Overlay Low Extremes (Blue)
    for date in extreme_days_low:
        if date in daily_profiles.index:
            plt.plot(daily_profiles.loc[date].index, daily_profiles.loc[date].values, 'b-', alpha=0.3)

    plt.title(f'Event Fingerprinting: Extreme Days vs Normal ({target_col})')
    plt.ylabel('MW')
    plt.xlabel('15-min Interval Index')
    plt.legend(['Normal Average', 'High Extremes', 'Low Extremes']) # Simple legend manual
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'event_fingerprints.png'))
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
            
            analyze_regime_segmentation(df, target)
            analyze_event_fingerprinting(df, target)
            
            print("\nPhase 3 Analysis Complete.")
        else:
            print("Error: Could not calculate Community Load.")
            
    except Exception as e:
        print(f"Error: {e}")
