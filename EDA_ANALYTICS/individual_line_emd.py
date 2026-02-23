import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyEMD import EMD
import os

# Configuration
INPUT_FILE = r'c:\Users\User\Desktop\project\llm_from_scratch-main\LOADFORECASINGPROJECT\resampled_data_15min.csv'
OUTPUT_BASE_DIR = r'c:\Users\User\Desktop\project\llm_from_scratch-main\LOADFORECASINGPROJECT\EDA_ANALYTICS\plots\individual_lines'
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

MW_COLUMNS = [
    'AD2NY_LINE (MW)', 'NY6ZA_LINE (MW)', 'NY3TU_LINE (MW)', 
    'BG1NY_LINE (MW)', 'BG4NY_LINE (MW)', '82T1_BANK (MW)', 
    '82T2_BANK (MW)', '82T3_BANK (MW)', '82T4_BANK (MW)', 
    '82T1YI (MW)', '82T2Y2 (MW)'
]

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    df['Datetime'] = pd.to_datetime(df['DATETIME'])
    df.set_index('Datetime', inplace=True)
    
    # Use a subset for speed (last 1000 points = ~10 days)
    df_subset = df.iloc[-1000:].copy() 
    return df_subset

def analyze_line(name, signal):
    print(f"Analyzing {name}...")
    # Fill NaNs if any
    signal = pd.Series(signal).interpolate(method='linear').fillna(0).values
    
    emd = EMD()
    imfs = emd.emd(signal)
    
    if len(imfs) == 0:
        print(f"Warning: No IMFs found for {name}. Signal might be too simple.")
        return {
            'Line': name,
            'Noise_Ratio': 0,
            'Num_IMFs': 0,
            'Peak_Load': np.max(np.abs(signal))
        }

    # Calculate noise metrics
    # Energy = sum of squares
    total_energy = np.sum(signal**2)
    imf_energies = [np.sum(imf**2) for imf in imfs]
    
    # IMF1 is usually high-frequency jitter/noise
    noise_energy = imf_energies[0] + (imf_energies[1] if len(imfs) > 1 else 0)
    noise_ratio = noise_energy / total_energy if total_energy > 0 else 0
    
    # Plotting
    n_plots = min(len(imfs), 4) + 1 # Original + first few IMFs + Residual
    plt.figure(figsize=(12, n_plots * 2))
    
    plt.subplot(n_plots, 1, 1)
    plt.plot(signal, 'black', label='Original')
    plt.title(f"EMD Decomposition: {name} (Noise Ratio: {noise_ratio:.2%})")
    plt.legend()
    
    for i in range(n_plots - 2):
        plt.subplot(n_plots, 1, i + 2)
        plt.plot(imfs[i], 'red', label=f'IMF {i+1}')
        plt.legend()
        
    # Residual (everything else)
    residual = signal - np.sum(imfs[:n_plots-2], axis=0) if len(imfs) > 0 else signal
    plt.subplot(n_plots, 1, n_plots)
    plt.plot(residual, 'blue', label='Clean Signal (Residual)')
    plt.legend()
    
    plt.tight_layout()
    safe_name = name.replace(' ', '_').replace('(', '').replace(')', '')
    plt.savefig(os.path.join(OUTPUT_BASE_DIR, f'{safe_name}_emd.png'))
    plt.close()
    
    return {
        'Line': name,
        'Noise_Ratio': noise_ratio,
        'Num_IMFs': len(imfs),
        'Peak_Load': np.max(np.abs(signal))
    }

def main():
    df = load_data(INPUT_FILE)
    results = []
    
    for col in MW_COLUMNS:
        if col in df.columns:
            res = analyze_line(col, df[col].values)
            results.append(res)
        else:
            print(f"Skipping {col}, not found in data.")
            
    # Save statistics
    stats_df = pd.DataFrame(results)
    stats_df = stats_df.sort_values(by='Noise_Ratio', ascending=False)
    stats_df.to_csv(os.path.join(OUTPUT_BASE_DIR, 'line_noise_stats.csv'), index=False)
    
    print("\nIndividual Line EMD Analysis Completed.")
    print(stats_df)

if __name__ == "__main__":
    main()
