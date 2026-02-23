import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
output_dir = 'plots/phase1'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])
    df.set_index('DATETIME', inplace=True)
    if 'DATE' in df.columns:
        df.drop(columns=['DATE'], inplace=True)
    return df

def check_sign_convention(df, target_cols):
    print("\n--- 1. Sign Convention Sanity Check ---")
    results = []
    for col in target_cols:
        positive_count = (df[col] > 0).sum()
        negative_count = (df[col] < 0).sum()
        zero_count = (df[col] == 0).sum()
        total = len(df)
        
        sign_type = "Mixed"
        if positive_count == total - zero_count: sign_type = "Positive Only"
        if negative_count == total - zero_count: sign_type = "Negative Only"
        
        results.append({
            'Column': col,
            'Sign Type': sign_type,
            '% Positive': round(positive_count/total*100, 2),
            '% Negative': round(negative_count/total*100, 2),
            '% Zero': round(zero_count/total*100, 2),
            'Min': df[col].min(),
            'Max': df[col].max(),
            'Mean': df[col].mean()
        })
    
    res_df = pd.DataFrame(results)
    print(res_df.to_string())
    res_df.to_csv(os.path.join(output_dir, 'sign_check.csv'), index=False)
    
    # Plot signs
    plt.figure(figsize=(12, 6))
    sns.barplot(data=res_df, x='Column', y='Mean')
    plt.xticks(rotation=45, ha='right')
    plt.title('Average Direction of Power Flow (Mean MW)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sign_convention_mean.png'))
    plt.close()

def check_flow_conservation(df, target_cols):
    print("\n--- 2. Flow Conservation Check ---")
    # Interpretation: Sum of all measured flows. 
    # If the substation is a node, Sum(In) + Sum(Out) ~ 0 (excluding losses).
    # We simply sum all signed columns.
    
    df['Net_Imbalance_MW'] = df[target_cols].sum(axis=1)
    
    stats = df['Net_Imbalance_MW'].describe()
    print("Net Imbalance Statistics:")
    print(stats)
    
    plt.figure(figsize=(15, 6))
    df['Net_Imbalance_MW'].plot(alpha=0.7, color='purple', linewidth=0.8)
    plt.title('Network Imbalance (Residual Flow) Over Time')
    plt.ylabel('Net MW')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'flow_conservation_imbalance.png'))
    plt.close()
    
    # Identify large imbalance periods (e.g., > 3 std dev)
    threshold = stats['mean'] + 3 * stats['std']
    anomalies = df[abs(df['Net_Imbalance_MW']) > abs(threshold)]
    print(f"\nIdentifying periods with imbalance > {threshold:.2f} MW:")
    if not anomalies.empty:
        print(f"Found {len(anomalies)} anomalous timestamps.")
        print(anomalies['Net_Imbalance_MW'].head(10).to_string())
    else:
        print("No extreme imbalance anomalies found.")

def check_missingness(df, target_cols):
    print("\n--- 3. Missingness and Outage Map ---")
    
    # Reindex to full frequency to capture time gaps
    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq='15T')
    df_reindexed = df.reindex(full_idx)
    
    total_gaps = df_reindexed[target_cols[0]].isnull().sum() # Assuming if one is missing, all are?
    print(f"Total missing timestamps after 15T reindexing: {total_gaps}")
    
    # Plot Heatmap of Data Availability (Binary)
    # 1 = Present, 0 = Missing
    availability = df_reindexed[target_cols].notnull().astype(int)
    
    plt.figure(figsize=(15, 8))
    sns.heatmap(availability.T, cmap='viridis', cbar=False, xticklabels=False)
    plt.title('Data Availability Heatmap (Yellow=Data, Purple=Missing)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'missingness_heatmap.png'))
    plt.close()
    
    if total_gaps > 0:
        print("\nTop missing windows:")
        is_missing = df_reindexed[target_cols[0]].isnull()
        # identifying blocks
        # This is a simple print of distinct missing chunks could be complex, omitting for brevity in logs
        pass

if __name__ == "__main__":
    file_path = 'resampled_data_15min.csv'
    try:
        df = load_data(file_path)
        target_cols = [col for col in df.columns if '(MW)' in col]
        print(f"Dimensions: {df.shape}")
        print(f"Target Cols: {target_cols}")
        
        check_sign_convention(df, target_cols)
        check_flow_conservation(df, target_cols)
        check_missingness(df, target_cols)
        
        print("\nPhase 1 Analysis Complete.")
        
    except Exception as e:
        print(f"Error: {e}")
