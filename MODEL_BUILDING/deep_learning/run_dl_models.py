"""
Deep Learning Model Training Pipeline (LSTM & GRU).

This script implements:
1. Data Loading & Scaling (StandardScaler)
2. Sliding Window Sequence Creation (Samples, Timesteps, Features)
3. PyTorch Model Definitions (LSTM, GRU)
4. Training Loop with Early Stopping
5. Evaluation against Test Set

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
import joblib
import matplotlib.pyplot as plt

# Add parent directory to path to import metrics
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.metrics import calculate_metrics, print_metrics

# --- Configuration ---
DATA_PATH = r"c:\Users\User\Desktop\project\llm_from_scratch-main\LOADFORECASINGPROJECT\FEATURE_ENGINEERING\outputs\engineered_features.csv"
OUTPUT_DIR = r"c:\Users\User\Desktop\project\llm_from_scratch-main\LOADFORECASINGPROJECT\MODEL_BUILDING\results"
MODEL_DIR = r"c:\Users\User\Desktop\project\llm_from_scratch-main\LOADFORECASINGPROJECT\MODEL_BUILDING\models"
PLOT_DIR = r"c:\Users\User\Desktop\project\llm_from_scratch-main\LOADFORECASINGPROJECT\MODEL_BUILDING\plots\deep_learning"

for d in [OUTPUT_DIR, MODEL_DIR, PLOT_DIR]:
    os.makedirs(d, exist_ok=True)

SPLIT_DATE = "2025-05-01"
TARGET_COL = "Community_Load_MW"
SEQ_LEN = 96  # 24 hours context (96 * 15min)
HORIZON = 1   # Predict next step (can extend to multi-step later)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_data():
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    df.index.name = 'Datetime'
    df = df.dropna()
    return df

class LoadDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_sequences(data, target, seq_len):
    """
    Create sliding window sequences.
    X: (Samples, Seq_Len, Features)
    y: (Samples, Horizon)
    """
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        x = data[i:(i + seq_len)]
        y = target[i + seq_len] # Next step prediction
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        out, _ = self.lstm(x, (h0, c0)) # out: (batch, seq_len, hidden)
        out = out[:, -1, :] # Take last time step
        out = self.fc(out)
        return out

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        out, _ = self.gru(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def train_model(model, train_loader, val_loader, epochs=50, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_loss = float('inf')
    patience = 10
    early_stop_counter = 0
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
            
        avg_train_loss = np.mean(batch_losses)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_batch_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                val_batch_losses.append(loss.item())
        
        avg_val_loss = np.mean(val_batch_losses)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"{model.__class__.__name__}.pth"))
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping triggered")
                break
                
    return train_losses, val_losses

def run_experiment():
    df = load_data()
    
    # 1. Feature Engineering / Selection
    drop_cols = ['Datetime'] # Keep Target in X for autoregression logic? 
    # Usually we scale features.
    
    # Split
    train_df = df[df.index < SPLIT_DATE]
    test_df = df[df.index >= SPLIT_DATE]
    
    # Scaling
    # Important: Fit scaler ONLY on train data
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    feature_cols = [c for c in df.columns if c not in drop_cols]
    target_col = TARGET_COL
    
    X_train_raw = train_df[feature_cols].values
    y_train_raw = train_df[[target_col]].values
    
    X_test_raw = test_df[feature_cols].values
    y_test_raw = test_df[[target_col]].values
    
    X_train_scaled = X_scaler.fit_transform(X_train_raw)
    y_train_scaled = y_scaler.fit_transform(y_train_raw)
    
    X_test_scaled = X_scaler.transform(X_test_raw)
    y_test_scaled = y_scaler.transform(y_test_raw)
    
    # Create Sequences
    print(f"Creating sequences with length {SEQ_LEN}...")
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled.flatten(), SEQ_LEN)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled.flatten(), SEQ_LEN)
    
    print(f"Train shapes: X={X_train_seq.shape}, y={y_train_seq.shape}")
    
    # Convert to Loader
    train_dataset = LoadDataset(X_train_seq, y_train_seq)
    test_dataset = LoadDataset(X_test_seq, y_test_seq)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    input_dim = X_train_seq.shape[2]
    hidden_dim = 64
    num_layers = 2
    output_dim = 1
    
    models_to_train = [
        ("LSTM", LSTMModel(input_dim, hidden_dim, num_layers, output_dim)),
        ("GRU", GRUModel(input_dim, hidden_dim, num_layers, output_dim))
    ]
    
    results = []
    
    for name, model in models_to_train:
        print(f"\n--- Training {name} ---")
        model.to(device)
        train_hist, val_hist = train_model(model, train_loader, test_loader)
        
        # Load best
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"{name}.pth")))
        model.eval()
        
        # Predict
        preds_scaled = []
        actuals_scaled = []
        with torch.no_grad():
            for X_b, y_b in test_loader:
                X_b = X_b.to(device)
                out = model(X_b)
                preds_scaled.extend(out.cpu().numpy().flatten())
                actuals_scaled.extend(y_b.numpy().flatten())
                
        # Inverse Scale
        preds = y_scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
        actuals = y_scaler.inverse_transform(np.array(actuals_scaled).reshape(-1, 1)).flatten()
        
        # Metrics
        metrics = calculate_metrics(pd.Series(actuals), pd.Series(preds))
        print_metrics(metrics, name)
        results.append({'Model': name, **metrics})
        
        # Plot
        plt.figure(figsize=(15, 6))
        plt.plot(train_hist, label='Train Loss')
        plt.plot(val_hist, label='Val Loss')
        plt.title(f'{name} Learning Curve')
        plt.legend()
        plt.savefig(os.path.join(PLOT_DIR, f'{name.lower()}_loss.png'))
        plt.close()
        
        plt.figure(figsize=(15, 6))
        plt.plot(actuals[:500], label='Actual', color='black')
        plt.plot(preds[:500], label='Predicted', color='red', alpha=0.7)
        plt.title(f'{name} Forecast vs Actual (Zoomed)')
        plt.legend()
        plt.savefig(os.path.join(PLOT_DIR, f'{name.lower()}_forecast.png'))
        plt.close()

    pd.DataFrame(results).to_csv(os.path.join(OUTPUT_DIR, 'dl_results.csv'), index=False)
    print("Deep Learning Phase Complete.")

if __name__ == "__main__":
    run_experiment()
