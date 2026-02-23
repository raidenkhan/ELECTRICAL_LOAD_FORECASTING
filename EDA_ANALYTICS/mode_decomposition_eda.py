import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyEMD import EMD
from vmdpy import VMD
import os
from scipy.fftpack import fft

# Configuration
INPUT_FILE = r'c:\Users\User\Desktop\project\llm_from_scratch-main\LOADFORECASINGPROJECT\resampled_data_15min.csv'
OUTPUT_DIR = r'c:\Users\User\Desktop\project\llm_from_scratch-main\LOADFORECASINGPROJECT\EDA_ANALYTICS\plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    # 'DATETIME' is the column name, need to specify it
    df = pd.read_csv(filepath)
    df['Datetime'] = pd.to_datetime(df['DATETIME'])
    df.set_index('Datetime', inplace=True)
    
    # Calculate Community Load
    load_cols = ['82T3_BANK (MW)', '82T4_BANK (MW)', '82T1_BANK (MW)']
    existing_cols = [c for c in load_cols if c in df.columns]
    
    if not existing_cols:
        raise ValueError("Could not find any of the expected load columns: " + str(load_cols))
        
    df['Load_MW'] = df[existing_cols].sum(axis=1)
    
    # Use a subset if the data is too large for EMD/VMD (e.g., last 2000 points or 1 month)
    # EMD is computationally expensive.
    df_subset = df.iloc[-2000:].copy() 
    
    # Fill missing values
    df_subset['Load_MW'] = df_subset['Load_MW'].interpolate(method='linear')
    return df_subset


def plot_imfs(signal, imfs, algorithm='EMD'):
    n_imfs = imfs.shape[0]
    plt.figure(figsize=(12, n_imfs * 2))
    
    # Plot original signal
    plt.subplot(n_imfs + 1, 1, 1)
    plt.plot(signal, 'r')
    plt.title(f"Original Signal & {algorithm} Decomposition")
    
    for i in range(n_imfs):
        plt.subplot(n_imfs + 1, 1, i + 2)
        plt.plot(imfs[i], 'g')
        plt.ylabel(f"IMF {i+1}")
        
        # Calculate mean period
        # Zero crossings
        zero_crossings = np.where(np.diff(np.sign(imfs[i])))[0]
        if len(zero_crossings) > 1:
            avg_period_samples = np.mean(np.diff(zero_crossings)) * 2
            avg_period_hours = (avg_period_samples * 15) / 60
            plt.title(f"IMF {i+1} (Mean Period: {avg_period_hours:.2f} hours)")
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, f'{algorithm.lower()}_decomposition.png')
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")
    plt.close()

def run_emd(signal_array):
    print("Running EMD...")
    emd = EMD()
    imfs = emd.emd(signal_array)
    return imfs

def run_vmd(signal_array):
    print("Running VMD...")
    # VMD parameters
    alpha = 2000       # moderate bandwidth constraint
    tau = 0.            # noise-tolerance (no strict fidelity enforcement)
    K = 5               # 3 modes -> trying to find daily, weekly, sub-daily
    DC = 0              # no DC part imposed
    init = 1            # initialize omegas uniformly
    tol = 1e-7
    
    # Run VMD
    u, u_hat, omega = VMD(signal_array, alpha, tau, K, DC, init, tol)
    return u

def main():
    try:
        df = load_data(INPUT_FILE)
        signal = df['Load_MW'].values
        
        # EMD
        imfs_emd = run_emd(signal)
        plot_imfs(signal, imfs_emd, algorithm='EMD')
        
        # VMD
        imfs_vmd = run_vmd(signal)
        plot_imfs(signal, imfs_vmd, algorithm='VMD')
        
        print("Mode Decomposition Analysis Completed.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
