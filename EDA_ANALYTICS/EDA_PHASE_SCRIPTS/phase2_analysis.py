import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
output_dir = 'plots/phase2'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])
    df.set_index('DATETIME', inplace=True)
    if 'DATE' in df.columns:
        df.drop(columns=['DATE'], inplace=True)
    return df

def analyze_intraday_stability(df, target_col):
    print("\n--- 4. Intraday Profile Stability ---")
    
    # Add calendar features temporarily
    df_feat = df[[target_col]].copy()
    df_feat['Hour'] = df_feat.index.hour
    df_feat['Minute'] = df_feat.index.minute
    df_feat['Month'] = df_feat.index.month
    df_feat['Time_Index'] = df_feat['Hour'] * 4 + df_feat['Minute'] / 15  # 0 to 95
    
    # Calculate Average Daily Profile by Month
    monthly_profiles = df_feat.groupby(['Month', 'Time_Index'])[target_col].mean().unstack(level=0)
    
    # Plot Profiles
    plt.figure(figsize=(12, 6))
    monthly_profiles.plot(ax=plt.gca(), alpha=0.7, linewidth=1.5)
    plt.title(f'Average Intraday Profile by Month ({target_col})')
    plt.ylabel('MW')
    plt.xlabel('15-min Interval Index (0-95)')
    plt.legend(title='Month', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'intraday_stability_monthly.png'))
    plt.close()
    
    # Measure Shape Similarity (Correlation between monthly profiles)
    corr_matrix = monthly_profiles.corr()
    print("Shape Similarity (Correlation between Monthly Profiles):")
    print(corr_matrix.round(2).to_string())
    
    avg_corr = corr_matrix.mean().mean()
    print(f"\nAverage Profile Correlation: {avg_corr:.3f}")

def analyze_weekday_weekend(df, target_col):
    print("\n--- 5. Weekday vs Weekend Decomposition ---")
    
    df_feat = df[[target_col]].copy()
    df_feat['DayOfWeek'] = df_feat.index.dayofweek
    df_feat['IsWeekend'] = df_feat['DayOfWeek'] >= 5
    df_feat['Time_Index'] = df_feat.index.hour * 4 + df_feat.index.minute / 15
    
    weekend_mask = df_feat['IsWeekend']
    weekday_mask = ~df_feat['IsWeekend']
    
    # Describe distributions
    print(f"Weekday Mean: {df_feat.loc[weekday_mask, target_col].mean():.2f} MW")
    print(f"Weekend Mean: {df_feat.loc[weekend_mask, target_col].mean():.2f} MW")
    
    # Plot Average Profiles
    avg_weekday = df_feat[weekday_mask].groupby('Time_Index')[target_col].mean()
    avg_weekend = df_feat[weekend_mask].groupby('Time_Index')[target_col].mean()
    
    plt.figure(figsize=(12, 6))
    avg_weekday.plot(label='Weekday (Mon-Fri)', color='blue', linewidth=2)
    avg_weekend.plot(label='Weekend (Sat-Sun)', color='orange', linewidth=2)
    plt.title(f'Weekday vs Weekend Average Load Profile ({target_col})')
    plt.ylabel('MW')
    plt.xlabel('15-min Interval Index')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'weekday_weekend_profile.png'))
    plt.close()
    
    # Boxplot of distribution
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df_feat['IsWeekend'], y=df_feat[target_col])
    plt.xticks([0, 1], ['Weekday', 'Weekend'])
    plt.title(f'Load Distribution: Weekday vs Weekend ({target_col})')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'weekday_weekend_boxplot.png'))
    plt.close()

def analyze_seasonal_drift(df, target_col):
    print("\n--- 6. Seasonal Drift in Baselines ---")
    
    # Resample to Daily or Monthly to see drift
    daily_mean = df[target_col].resample('D').mean()
    rolling_30d = df[target_col].rolling(window=30*24*4).mean() # 30 days
    
    plt.figure(figsize=(15, 6))
    daily_mean.plot(alpha=0.5, label='Daily Mean', color='gray')
    rolling_30d.plot(linewidth=2, label='30-Day Rolling Mean', color='red')
    plt.title(f'Seasonal Drift Analysis ({target_col})')
    plt.ylabel('MW')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'seasonal_drift.png'))
    plt.close()
    
    # Calculate Monthly Medians for Table
    monthly_medians = df[target_col].resample('ME').median() # 'M' is deprecated for Month End in newer pandas, usage varies. using 'M' usually safe or 'ME'
    print("\nMonthly Medians:")
    print(monthly_medians.tail(12).to_string())

if __name__ == "__main__":
    file_path = 'resampled_data_15min.csv'
    try:
        df = load_data(file_path)
        
        # Calculate Community Load (The Target)
        load_cols = ['82T3_BANK (MW)', '82T4_BANK (MW)', '82T1_BANK (MW)']
        existing_cols = [c for c in load_cols if c in df.columns]
        
        if existing_cols:
            df['Community_Load_MW'] = df[existing_cols].sum(axis=1)
            target = 'Community_Load_MW'
            print(f"Analyzing Target: {target}")
            
            analyze_intraday_stability(df, target)
            analyze_weekday_weekend(df, target)
            analyze_seasonal_drift(df, target)
            
            print("\nPhase 2 Analysis Complete.")
        else:
            print("Error: Could not calculate Community Load. Missing columns.")
            
    except Exception as e:
        print(f"Error: {e}")
