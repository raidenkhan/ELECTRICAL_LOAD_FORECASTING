import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load the data
print("Loading data...")
df = pd.read_csv('../resampled_data_15min.csv')
df['DATETIME'] = pd.to_datetime(df['DATETIME'])
df.set_index('DATETIME', inplace=True)

# Extract frequency column
freq_col = 'FREQ (HZ)'

print("\n" + "="*80)
print("DETAILED FREQUENCY ANALYSIS - POWER SYSTEM DATASET")
print("="*80)

# ============================================================================
# 1. BASIC STATISTICS
# ============================================================================
print("\n1. BASIC STATISTICS")
print("-" * 80)
print(f"Total Records: {len(df):,}")
print(f"Date Range: {df.index.min()} to {df.index.max()}")
print(f"Duration: {(df.index.max() - df.index.min()).days} days")
print(f"\nFrequency Statistics:")
print(f"  Mean:     {df[freq_col].mean():.6f} Hz")
print(f"  Median:   {df[freq_col].median():.6f} Hz")
print(f"  Std Dev:  {df[freq_col].std():.6f} Hz")
print(f"  Min:      {df[freq_col].min():.6f} Hz")
print(f"  Max:      {df[freq_col].max():.6f} Hz")
print(f"  Range:    {df[freq_col].max() - df[freq_col].min():.6f} Hz")
print(f"  CV:       {(df[freq_col].std() / df[freq_col].mean() * 100):.4f}%")

# ============================================================================
# 2. FREQUENCY DEVIATION ANALYSIS
# ============================================================================
print("\n2. FREQUENCY DEVIATION ANALYSIS")
print("-" * 80)
nominal_freq = 50.0  # Hz
df['freq_deviation'] = df[freq_col] - nominal_freq
df['freq_deviation_pct'] = (df['freq_deviation'] / nominal_freq) * 100

print(f"Nominal Frequency: {nominal_freq} Hz")
print(f"\nDeviation Statistics:")
print(f"  Mean Deviation:     {df['freq_deviation'].mean():.6f} Hz ({df['freq_deviation_pct'].mean():.4f}%)")
print(f"  Std Dev Deviation:  {df['freq_deviation'].std():.6f} Hz")
print(f"  Max Positive Dev:   +{df['freq_deviation'].max():.6f} Hz ({df['freq_deviation_pct'].max():.4f}%)")
print(f"  Max Negative Dev:   {df['freq_deviation'].min():.6f} Hz ({df['freq_deviation_pct'].min():.4f}%)")

# ============================================================================
# 3. GRID CODE COMPLIANCE ANALYSIS
# ============================================================================
print("\n3. GRID CODE COMPLIANCE ANALYSIS")
print("-" * 80)

# Define frequency limits (typical grid codes)
limits = {
    'Normal Operation': (49.8, 50.2),
    'Acceptable Range': (49.5, 50.5),
    'Critical Low': (49.0, 49.5),
    'Critical High': (50.5, 51.0),
    'Emergency': (None, None)  # Outside all ranges
}

for category, (low, high) in limits.items():
    if category == 'Emergency':
        count = len(df[(df[freq_col] < 49.0) | (df[freq_col] > 51.0)])
    else:
        count = len(df[(df[freq_col] >= low) & (df[freq_col] < high)])
    percentage = (count / len(df)) * 100
    print(f"  {category:20s}: {count:6,} records ({percentage:6.2f}%)")

# ============================================================================
# 4. FREQUENCY STABILITY METRICS
# ============================================================================
print("\n4. FREQUENCY STABILITY METRICS")
print("-" * 80)

# Rate of Change of Frequency (ROCOF)
df['rocof'] = df[freq_col].diff() / (15/60)  # Hz/minute (15-min intervals)
df['rocof_abs'] = df['rocof'].abs()

print(f"Rate of Change of Frequency (ROCOF):")
print(f"  Mean ROCOF:        {df['rocof'].mean():.6f} Hz/min")
print(f"  Mean |ROCOF|:      {df['rocof_abs'].mean():.6f} Hz/min")
print(f"  Max ROCOF:         {df['rocof'].max():.6f} Hz/min")
print(f"  Min ROCOF:         {df['rocof'].min():.6f} Hz/min")
print(f"  Std Dev ROCOF:     {df['rocof'].std():.6f} Hz/min")

# Frequency excursions
excursion_threshold = 0.2  # Hz
excursions = df[df['freq_deviation'].abs() > excursion_threshold]
print(f"\nFrequency Excursions (>{excursion_threshold} Hz deviation):")
print(f"  Total Excursions:  {len(excursions):,} ({len(excursions)/len(df)*100:.2f}%)")

# ============================================================================
# 5. TEMPORAL PATTERNS
# ============================================================================
print("\n5. TEMPORAL PATTERNS")
print("-" * 80)

# Add time features
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month

# Hourly patterns
hourly_stats = df.groupby('hour')[freq_col].agg(['mean', 'std', 'min', 'max'])
print("\nHourly Frequency Patterns:")
print(f"  Most stable hour:   {hourly_stats['std'].idxmin()}:00 (Std: {hourly_stats['std'].min():.6f} Hz)")
print(f"  Most variable hour: {hourly_stats['std'].idxmax()}:00 (Std: {hourly_stats['std'].max():.6f} Hz)")
print(f"  Highest avg freq:   {hourly_stats['mean'].idxmax()}:00 ({hourly_stats['mean'].max():.6f} Hz)")
print(f"  Lowest avg freq:    {hourly_stats['mean'].idxmin()}:00 ({hourly_stats['mean'].min():.6f} Hz)")

# Daily patterns
daily_stats = df.groupby('day_of_week')[freq_col].agg(['mean', 'std'])
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
print("\nDaily Frequency Patterns:")
for day_idx, day_name in enumerate(days):
    if day_idx in daily_stats.index:
        print(f"  {day_name:10s}: Mean={daily_stats.loc[day_idx, 'mean']:.6f} Hz, Std={daily_stats.loc[day_idx, 'std']:.6f} Hz")

# ============================================================================
# 6. DISTRIBUTION ANALYSIS
# ============================================================================
print("\n6. DISTRIBUTION ANALYSIS")
print("-" * 80)

# Normality tests
shapiro_stat, shapiro_p = stats.shapiro(df[freq_col].sample(min(5000, len(df))))
ks_stat, ks_p = stats.kstest(df[freq_col], 'norm', args=(df[freq_col].mean(), df[freq_col].std()))

print(f"Normality Tests:")
print(f"  Shapiro-Wilk: statistic={shapiro_stat:.6f}, p-value={shapiro_p:.6e}")
print(f"  Kolmogorov-Smirnov: statistic={ks_stat:.6f}, p-value={ks_p:.6e}")
print(f"  Distribution: {'Normal' if shapiro_p > 0.05 else 'Non-Normal'}")

# Skewness and Kurtosis
print(f"\nDistribution Shape:")
print(f"  Skewness: {stats.skew(df[freq_col]):.6f}")
print(f"  Kurtosis: {stats.kurtosis(df[freq_col]):.6f}")

# Percentiles
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
print(f"\nPercentiles:")
for p in percentiles:
    val = np.percentile(df[freq_col], p)
    print(f"  {p:2d}th: {val:.6f} Hz")

# ============================================================================
# 7. FREQUENCY EVENTS ANALYSIS
# ============================================================================
print("\n7. FREQUENCY EVENTS ANALYSIS")
print("-" * 80)

# Define event thresholds
low_freq_threshold = 49.8
high_freq_threshold = 50.2

low_freq_events = df[df[freq_col] < low_freq_threshold]
high_freq_events = df[df[freq_col] > high_freq_threshold]

print(f"Low Frequency Events (<{low_freq_threshold} Hz):")
print(f"  Count: {len(low_freq_events):,} ({len(low_freq_events)/len(df)*100:.2f}%)")
if len(low_freq_events) > 0:
    print(f"  Lowest: {low_freq_events[freq_col].min():.6f} Hz at {low_freq_events[freq_col].idxmin()}")
    print(f"  Mean: {low_freq_events[freq_col].mean():.6f} Hz")

print(f"\nHigh Frequency Events (>{high_freq_threshold} Hz):")
print(f"  Count: {len(high_freq_events):,} ({len(high_freq_events)/len(df)*100:.2f}%)")
if len(high_freq_events) > 0:
    print(f"  Highest: {high_freq_events[freq_col].max():.6f} Hz at {high_freq_events[freq_col].idxmax()}")
    print(f"  Mean: {high_freq_events[freq_col].mean():.6f} Hz")

# ============================================================================
# 8. AUTOCORRELATION ANALYSIS
# ============================================================================
print("\n8. AUTOCORRELATION ANALYSIS")
print("-" * 80)

# Calculate autocorrelation for different lags
lags = [1, 4, 24, 96]  # 15min, 1hr, 6hrs, 24hrs
print("Autocorrelation at different lags:")
for lag in lags:
    if lag < len(df):
        autocorr = df[freq_col].autocorr(lag=lag)
        lag_time = lag * 15  # minutes
        print(f"  Lag {lag:3d} ({lag_time:4d} min): {autocorr:.6f}")

# ============================================================================
# 9. MISSING DATA ANALYSIS
# ============================================================================
print("\n9. DATA QUALITY")
print("-" * 80)
missing_count = df[freq_col].isna().sum()
print(f"Missing Values: {missing_count:,} ({missing_count/len(df)*100:.2f}%)")
print(f"Zero Values: {(df[freq_col] == 0).sum():,}")
print(f"Duplicate Timestamps: {df.index.duplicated().sum():,}")

# ============================================================================
# 10. VISUALIZATION
# ============================================================================
print("\n10. GENERATING VISUALIZATIONS...")
print("-" * 80)

fig = plt.figure(figsize=(20, 24))

# 1. Time series plot
ax1 = plt.subplot(6, 2, 1)
df[freq_col].plot(ax=ax1, linewidth=0.5, alpha=0.7)
ax1.axhline(y=50, color='r', linestyle='--', label='Nominal (50 Hz)')
ax1.axhline(y=49.8, color='orange', linestyle='--', alpha=0.5, label='Normal Range')
ax1.axhline(y=50.2, color='orange', linestyle='--', alpha=0.5)
ax1.set_title('Frequency Time Series', fontsize=14, fontweight='bold')
ax1.set_ylabel('Frequency (Hz)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Frequency deviation
ax2 = plt.subplot(6, 2, 2)
df['freq_deviation'].plot(ax=ax2, linewidth=0.5, alpha=0.7, color='red')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax2.set_title('Frequency Deviation from Nominal', fontsize=14, fontweight='bold')
ax2.set_ylabel('Deviation (Hz)')
ax2.grid(True, alpha=0.3)

# 3. Distribution histogram
ax3 = plt.subplot(6, 2, 3)
df[freq_col].hist(bins=100, ax=ax3, edgecolor='black', alpha=0.7)
ax3.axvline(x=50, color='r', linestyle='--', linewidth=2, label='Nominal')
ax3.axvline(x=df[freq_col].mean(), color='g', linestyle='--', linewidth=2, label='Mean')
ax3.set_title('Frequency Distribution', fontsize=14, fontweight='bold')
ax3.set_xlabel('Frequency (Hz)')
ax3.set_ylabel('Count')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Box plot
ax4 = plt.subplot(6, 2, 4)
df[freq_col].plot(kind='box', ax=ax4)
ax4.axhline(y=50, color='r', linestyle='--', label='Nominal')
ax4.set_title('Frequency Box Plot', fontsize=14, fontweight='bold')
ax4.set_ylabel('Frequency (Hz)')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Hourly patterns
ax5 = plt.subplot(6, 2, 5)
hourly_stats['mean'].plot(ax=ax5, marker='o', linewidth=2)
ax5.fill_between(hourly_stats.index, 
                  hourly_stats['mean'] - hourly_stats['std'],
                  hourly_stats['mean'] + hourly_stats['std'],
                  alpha=0.3)
ax5.axhline(y=50, color='r', linestyle='--', label='Nominal')
ax5.set_title('Hourly Frequency Pattern', fontsize=14, fontweight='bold')
ax5.set_xlabel('Hour of Day')
ax5.set_ylabel('Frequency (Hz)')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Daily patterns
ax6 = plt.subplot(6, 2, 6)
daily_stats['mean'].plot(kind='bar', ax=ax6, color='steelblue', edgecolor='black')
ax6.axhline(y=50, color='r', linestyle='--', label='Nominal')
ax6.set_title('Daily Frequency Pattern', fontsize=14, fontweight='bold')
ax6.set_xlabel('Day of Week')
ax6.set_ylabel('Mean Frequency (Hz)')
ax6.set_xticklabels(days, rotation=45)
ax6.legend()
ax6.grid(True, alpha=0.3)

# 7. ROCOF
ax7 = plt.subplot(6, 2, 7)
df['rocof'].plot(ax=ax7, linewidth=0.5, alpha=0.5)
ax7.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax7.set_title('Rate of Change of Frequency (ROCOF)', fontsize=14, fontweight='bold')
ax7.set_ylabel('ROCOF (Hz/min)')
ax7.grid(True, alpha=0.3)

# 8. ROCOF distribution
ax8 = plt.subplot(6, 2, 8)
df['rocof'].dropna().hist(bins=100, ax=ax8, edgecolor='black', alpha=0.7)
ax8.axvline(x=0, color='r', linestyle='--', linewidth=2)
ax8.set_title('ROCOF Distribution', fontsize=14, fontweight='bold')
ax8.set_xlabel('ROCOF (Hz/min)')
ax8.set_ylabel('Count')
ax8.grid(True, alpha=0.3)

# 9. QQ plot
ax9 = plt.subplot(6, 2, 9)
stats.probplot(df[freq_col], dist="norm", plot=ax9)
ax9.set_title('Q-Q Plot (Normality Check)', fontsize=14, fontweight='bold')
ax9.grid(True, alpha=0.3)

# 10. Cumulative distribution
ax10 = plt.subplot(6, 2, 10)
sorted_freq = np.sort(df[freq_col])
cumulative = np.arange(1, len(sorted_freq) + 1) / len(sorted_freq) * 100
ax10.plot(sorted_freq, cumulative, linewidth=2)
ax10.axvline(x=50, color='r', linestyle='--', label='Nominal')
ax10.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
ax10.set_title('Cumulative Distribution Function', fontsize=14, fontweight='bold')
ax10.set_xlabel('Frequency (Hz)')
ax10.set_ylabel('Cumulative Percentage (%)')
ax10.legend()
ax10.grid(True, alpha=0.3)

# 11. Heatmap - Hour vs Day of Week
ax11 = plt.subplot(6, 2, 11)
pivot_table = df.pivot_table(values=freq_col, index='hour', columns='day_of_week', aggfunc='mean')
sns.heatmap(pivot_table, cmap='RdYlGn', center=50, ax=ax11, cbar_kws={'label': 'Frequency (Hz)'})
ax11.set_title('Frequency Heatmap: Hour vs Day of Week', fontsize=14, fontweight='bold')
ax11.set_xlabel('Day of Week')
ax11.set_ylabel('Hour of Day')
ax11.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

# 12. Monthly trend
ax12 = plt.subplot(6, 2, 12)
monthly_stats = df.groupby('month')[freq_col].agg(['mean', 'std'])
monthly_stats['mean'].plot(ax=ax12, marker='o', linewidth=2, color='darkblue')
ax12.fill_between(monthly_stats.index,
                   monthly_stats['mean'] - monthly_stats['std'],
                   monthly_stats['mean'] + monthly_stats['std'],
                   alpha=0.3)
ax12.axhline(y=50, color='r', linestyle='--', label='Nominal')
ax12.set_title('Monthly Frequency Pattern', fontsize=14, fontweight='bold')
ax12.set_xlabel('Month')
ax12.set_ylabel('Frequency (Hz)')
ax12.set_xticks(range(1, 13))
ax12.legend()
ax12.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('frequency_eda_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: frequency_eda_analysis.png")

# ============================================================================
# Additional focused plots
# ============================================================================

# Compliance pie chart
fig2, axes = plt.subplots(2, 2, figsize=(16, 12))

# Compliance distribution
ax_pie = axes[0, 0]
compliance_data = []
compliance_labels = []
for category, (low, high) in limits.items():
    if category == 'Emergency':
        count = len(df[(df[freq_col] < 49.0) | (df[freq_col] > 51.0)])
    else:
        count = len(df[(df[freq_col] >= low) & (df[freq_col] < high)])
    if count > 0:
        compliance_data.append(count)
        compliance_labels.append(f"{category}\n({count:,}, {count/len(df)*100:.1f}%)")

colors = ['green', 'lightgreen', 'orange', 'red', 'darkred'][:len(compliance_data)]
ax_pie.pie(compliance_data, labels=compliance_labels, colors=colors, autopct='', startangle=90)
ax_pie.set_title('Grid Code Compliance Distribution', fontsize=14, fontweight='bold')

# Violin plot by hour
ax_violin = axes[0, 1]
hour_data = [df[df['hour'] == h][freq_col].values for h in range(24)]
parts = ax_violin.violinplot(hour_data, positions=range(24), showmeans=True, showmedians=True)
ax_violin.axhline(y=50, color='r', linestyle='--', label='Nominal')
ax_violin.set_title('Frequency Distribution by Hour (Violin Plot)', fontsize=14, fontweight='bold')
ax_violin.set_xlabel('Hour of Day')
ax_violin.set_ylabel('Frequency (Hz)')
ax_violin.set_xticks(range(0, 24, 2))
ax_violin.legend()
ax_violin.grid(True, alpha=0.3)

# Scatter plot: Frequency vs ROCOF
ax_scatter = axes[1, 0]
sample_size = min(10000, len(df))
sample_df = df.sample(sample_size)
scatter = ax_scatter.scatter(sample_df[freq_col], sample_df['rocof'], 
                             c=sample_df['hour'], cmap='viridis', alpha=0.5, s=10)
ax_scatter.axvline(x=50, color='r', linestyle='--', alpha=0.5)
ax_scatter.axhline(y=0, color='r', linestyle='--', alpha=0.5)
ax_scatter.set_title('Frequency vs ROCOF (colored by hour)', fontsize=14, fontweight='bold')
ax_scatter.set_xlabel('Frequency (Hz)')
ax_scatter.set_ylabel('ROCOF (Hz/min)')
plt.colorbar(scatter, ax=ax_scatter, label='Hour of Day')
ax_scatter.grid(True, alpha=0.3)

# Rolling statistics
ax_rolling = axes[1, 1]
rolling_mean = df[freq_col].rolling(window=96).mean()  # 24-hour rolling mean
rolling_std = df[freq_col].rolling(window=96).std()
ax_rolling.plot(rolling_mean.index, rolling_mean, label='24h Rolling Mean', linewidth=1.5)
ax_rolling.fill_between(rolling_mean.index,
                         rolling_mean - 2*rolling_std,
                         rolling_mean + 2*rolling_std,
                         alpha=0.3, label='±2σ')
ax_rolling.axhline(y=50, color='r', linestyle='--', label='Nominal')
ax_rolling.set_title('24-Hour Rolling Statistics', fontsize=14, fontweight='bold')
ax_rolling.set_ylabel('Frequency (Hz)')
ax_rolling.legend()
ax_rolling.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('frequency_eda_detailed.png', dpi=300, bbox_inches='tight')
print("Saved: frequency_eda_detailed.png")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("\nGenerated Files:")
print("  1. frequency_eda_analysis.png - Comprehensive 12-panel analysis")
print("  2. frequency_eda_detailed.png - Detailed 4-panel analysis")
print("\nKey Findings Summary:")
print(f"  • Frequency operates around {df[freq_col].mean():.4f} Hz (nominal: 50 Hz)")
print(f"  • Deviation range: {df['freq_deviation'].min():.4f} to {df['freq_deviation'].max():.4f} Hz")
print(f"  • Grid compliance: {len(df[(df[freq_col] >= 49.8) & (df[freq_col] <= 50.2)])/len(df)*100:.2f}% within normal range")
print(f"  • Mean ROCOF: {df['rocof_abs'].mean():.6f} Hz/min")
print(f"  • Most stable hour: {hourly_stats['std'].idxmin()}:00")
print(f"  • Most variable hour: {hourly_stats['std'].idxmax()}:00")
