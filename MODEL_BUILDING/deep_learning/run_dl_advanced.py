"""
Deep Learning Training Pipeline: LSTM & TCN.

Objectives:
1. Train LSTM and TCN models.
2. Evaluate on 6-hour horizon (Multi-Output/Seq2Seq).
3. Compare against Seasonal Naive @ 6h.
4. Estimate Uncertainty (Monte Carlo Dropout).

Author: Load Forecasting Team
"""

import sys
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import joblib

# Configuration
DATA_PATH = r"c:\Users\User\Desktop\project\llm_from_scratch-main\LOADFORECASINGPROJECT\FEATURE_ENGINEERING\outputs\engineered_features.csv"
OUTPUT_DIR = r"c:\Users\User\Desktop\project\llm_from_scratch-main\LOADFORECASINGPROJECT\MODEL_BUILDING\results_dl"
MODEL_DIR = r"c:\Users\User\Desktop\project\llm_from_scratch-main\LOADFORECASINGPROJECT\MODEL_BUILDING\models"
PLOT_DIR = r"c:\Users\User\Desktop\project\llm_from_scratch-main\LOADFORECASINGPROJECT\MODEL_BUILDING\plots\deep_learning"

for d in [OUTPUT_DIR, MODEL_DIR, PLOT_DIR]:
    os.makedirs(d, exist_ok=True)

SPLIT_DATE = "2025-05-01"
TARGET_COL = "Community_Load_MW"
SEQ_LEN = 96       # 24 hours context
HORIZON = 24       # 6 hours prediction (24 * 15min)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data():
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    df.index.name = 'Datetime'
    df = df.dropna()
    return df

# --- Datasets ---
class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def create_sequences(data, target, seq_len, horizon):
    xs, ys = [], []
    # Ensure we don't go out of bounds
    for i in range(len(data) - seq_len - horizon + 1):
        x = data[i : i+seq_len]
        y = target[i+seq_len : i+seq_len+horizon]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# --- Models ---
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers, output_dim, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)
        # We want to predict a sequence of length HORIZON. 
        # For simple Seq2Seq in this script, we'll map the *last* hidden state to the entire output vector.
        # Structure: Many-to-Many (Encoder-Decoder) is better, but Many-to-One-Vector is simpler standard baseline.
        last_hidden = out[:, -1, :] 
        out = self.fc(last_hidden) 
        return out

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # Causal Padding: Input padded effectively on the left
        # We handle padding in forward to ensure it's Causal (only past data)
        self.padding = padding
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=0, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=0, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1, self.conv2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        # Causal Padding for Conv1
        out = torch.nn.functional.pad(x, (self.padding, 0))
        out = self.conv1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        # Causal Padding for Conv2
        out = torch.nn.functional.pad(out, (self.padding, 0))
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_channels, kernel_size=3, dropout=0.2):
        super(TCNModel, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_dim)

    def forward(self, x):
        # x: (batch, seq_len, features) -> TCN needs (batch, features, seq_len)
        x = x.transpose(1, 2)
        y = self.tcn(x)
        # Back to (batch, seq_len, features)
        y = y.transpose(1, 2)
        # Take last time step for prediction
        y = y[:, -1, :]
        return self.fc(y)

# --- Uncertainty with Monte Carlo Dropout ---
def predict_with_uncertainty(model, loader, n_samples=20):
    model.train() # Enable dropout
    preds_list = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            batch_preds = []
            for X_b, _ in loader:
                X_b = X_b.to(DEVICE)
                out = model(X_b)
                batch_preds.append(out.cpu().numpy())
            preds_list.append(np.concatenate(batch_preds, axis=0))
            
    # shape: (n_samples, n_test, horizon)
    preds_stack = np.stack(preds_list)
    mean_pred = preds_stack.mean(axis=0)
    std_pred = preds_stack.std(axis=0)
    
    # Calculate P10, P90
    p10 = np.percentile(preds_stack, 10, axis=0)
    p90 = np.percentile(preds_stack, 90, axis=0)
    
    return mean_pred, std_pred, p10, p90

def train_and_eval():
    df = load_data()
    print(f"Data Loaded: {df.shape}")
    
    # Preprocessing
    feature_cols = [c for c in df.columns if c != 'Datetime'] # Keep Target in X? Yes usually
    target_col = TARGET_COL
    
    # Split
    train_df = df[df.index < SPLIT_DATE]
    test_df = df[df.index >= SPLIT_DATE]
    
    # Scaling
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_s = scaler_X.fit_transform(train_df[feature_cols])
    y_train_s = scaler_y.fit_transform(train_df[[target_col]])
    
    X_test_s = scaler_X.transform(test_df[feature_cols])
    y_test_s = scaler_y.transform(test_df[[target_col]])
    
    # Sequences
    X_train_seq, y_train_seq = create_sequences(X_train_s, y_train_s.flatten(), SEQ_LEN, HORIZON)
    X_test_seq, y_test_seq = create_sequences(X_test_s, y_test_s.flatten(), SEQ_LEN, HORIZON)
    
    print(f"Train Seq: {X_train_seq.shape}", flush=True)
    
    train_dataset = SequenceDataset(X_train_seq, y_train_seq)
    test_dataset = SequenceDataset(X_test_seq, y_test_seq)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    input_dim = X_train_s.shape[1]
    
    models = {
        'LSTM': LSTMModel(input_dim, 64, 2, HORIZON).to(DEVICE),
        'TCN': TCNModel(input_dim, HORIZON, [32, 32, 32]).to(DEVICE)
    }
    
    results = {}
    
    # 1. Seasonal Naive @ 6h Calculation
    # Seasonal Naive: Forecast(t) = Actual(t - 24h)
    # For a horizon of 6h (t+1..t+24), we use (t+1-96 .. t+24-96)
    # Which corresponds to yesterday at the same times.
    
    # Get test targets (unscaled)
    # y_test_unscaled = test_df[[target_col]].values[SEQ_LEN : SEQ_LEN+len(y_test_seq)+HORIZON-1] 
    # That indexing is tricky. Let's use the created y_test_seq sequences and inverse transform
    y_test_true = scaler_y.inverse_transform(y_test_seq.reshape(-1, HORIZON)) # (Samples, 24)
    
    # Construct Seasonal Naive Forecasts
    # We need inputs corresponding to yesterday.
    # X_test_df contains the raw data.
    # For sample i (time t), target is t+1...t+24.
    # Seasonal Naive pred is Actual(t+1-96)...Actual(t+24-96).
    # This roughly equals: values at X[i-96+1 ... ]
    # Easier: Just shift the whole Series by 96 (24h) and extract windows.
    
    full_target_series = df[target_col]
    shifted_series = full_target_series.shift(96) # Value at t is now Value at t-96
    
    # Align with test samples
    # The first test sample predicts [SPLIT_DATE + 15min ... + 6h]
    # We want [SPLIT_DATE + 15min - 24h ... ]
    test_start_idx = len(train_df)
    
    sn_preds = []
    # Loop matches create_sequences logic
    for i in range(len(test_df) - SEQ_LEN - HORIZON + 1):
        # The index in the full dataframe is test_start_idx + i
        # Target start: test_start_idx + i + seq_len
        current_t = test_start_idx + i + SEQ_LEN
        
        # We want the 24 steps that happened 96 steps before these targets
        # Target indices: current_t ... current_t + HORIZON
        # SN indices: current_t - 96 ... current_t + HORIZON - 96
        
        # Check bounds
        sn_window = full_target_series.iloc[current_t - 96 : current_t + HORIZON - 96].values
        sn_preds.append(sn_window)
        
    sn_preds = np.array(sn_preds)
    
    # Calculate SN MAE @ 6h
    # Horizon 24 is index 23
    sn_mae_6h = np.mean(np.abs(y_test_true[:, -1] - sn_preds[:, -1]))
    print(f"Seasonal Naive MAE @ 6h: {sn_mae_6h:.2f} MW", flush=True)
    results['Seasonal_Naive'] = {'mae_6h': sn_mae_6h}
    
    # Training Loop
    criterion = nn.MSELoss()
    
    for name, model in models.items():
        print(f"\n--- Training {name} ---", flush=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(5): # Reduced for speed comparison
            model.train()
            losses = []
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                optimizer.zero_grad()
                out = model(X_b)
                loss = criterion(out, y_b)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            print(f"Epoch {epoch}: {np.mean(losses):.4f}", flush=True)
        
        # Uncertainty Prediction
        mean_pred_s, std_pred_s, p10_s, p90_s = predict_with_uncertainty(model, test_loader)
        
        # Inverse transform
        mean_pred = scaler_y.inverse_transform(mean_pred_s)
        p10 = scaler_y.inverse_transform(p10_s)
        p90 = scaler_y.inverse_transform(p90_s)
        
        # Calculate MAE @ 6h
        mae_6h = mean_absolute_error(y_test_true[:, -1], mean_pred[:, -1])
        print(f"{name} MAE @ 6h: {mae_6h:.2f} MW", flush=True)
        
        # Coverage
        coverage = np.mean((y_test_true >= p10) & (y_test_true <= p90)) * 100
        print(f"{name} Uncertainty Coverage: {coverage:.2f}%", flush=True)
        
        results[name] = {
            'mae_6h': mae_6h, 
            'coverage': coverage,
            'preds': mean_pred,
            'p10': p10, 
            'p90': p90
        }
        
        # Plot
        plt.figure(figsize=(10, 6))
        # Plot first 100 samples, horizon step 6h
        plt.plot(y_test_true[:100, -1], label='Actual (6h ahead)', color='black')
        plt.plot(mean_pred[:100, -1], label=f'{name} Mean', color='blue')
        plt.fill_between(range(100), p10[:100, -1], p90[:100, -1], color='blue', alpha=0.2, label='Uncertainty')
        plt.title(f'{name} 6-Hour Ahead Forecast (with Uncertainty)')
        plt.legend()
        plt.savefig(os.path.join(PLOT_DIR, f'{name}_6h_forecast.png'))
        plt.close()

    # Save Results
    pd.DataFrame(results).to_json(os.path.join(OUTPUT_DIR, 'dl_metrics.json'))
    print("Completed.", flush=True)

if __name__ == "__main__":
    train_and_eval()
